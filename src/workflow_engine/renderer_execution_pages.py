"""Direct-agent execution page rendering, caching, and progress card pagination."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

from src.card.render.budget import RenderBudget
from src.card.render.renderer import render_card as _default_render_card
from src.card.shared.text_safety import (
    neutralize_feishu_rich_text_controls,
    sanitize_card_text_for_audit,
    sanitize_markdown_image_references,
)
from src.card.shared.truncation import (
    FEISHU_CARD_TABLE_LIMIT,
    count_markdown_table_blocks,
    count_tagged_nodes,
    normalize_markdown_tables_for_card,
)
from src.card.state.models import (
    CardMetadata,
    CardState,
    ContentBlock,
    HeaderState,
    TerminalStatus,
)
from src.card.state.runtime_stats import RuntimeStats

from .models import AgentProgress, AgentStatus, WorkflowProject
from .renderer_elements import (
    _WORKFLOW_FRAGMENT_MAX_BYTES,
    _WORKFLOW_FRAGMENT_MAX_NODES,
    _binding_model_parts,
    _card_text_for_agent_output,
    _resolve_renderer_attr,
    _safe_dynamic_text,
    _set_workflow_page_key,
    _utf8_safe_text,
)


def _ordered_direct_agents(project: WorkflowProject) -> list[AgentProgress]:
    """Return all direct-call agents sorted by invocation order."""
    flattened = [agent for phase in project.phases for agent in phase.agents]
    return [
        agent
        for _, agent in sorted(
            enumerate(flattened),
            key=lambda pair: (
                int(getattr(pair[1], "call_index", pair[0]) or 0),
                pair[0],
            ),
        )
    ]


def _identified_direct_agents(
    project: WorkflowProject,
) -> list[tuple[AgentProgress, str]]:
    """Pair calls with stable identities independent of mutable list position."""
    seen: dict[str, int] = {}
    identified: list[tuple[AgentProgress, str]] = []
    for agent in _ordered_direct_agents(project):
        identity_payload = json.dumps(
            {
                "agent_id": str(getattr(agent, "agent_id", None) or ""),
                "call_index": int(getattr(agent, "call_index", 0) or 0),
                "effort": str(getattr(agent, "effort", None) or ""),
                "label": str(getattr(agent, "label", "") or ""),
                "model": str(getattr(agent, "model", None) or ""),
                "profile": str(getattr(agent, "profile", None) or ""),
                "role": str(getattr(agent, "role", None) or ""),
                "started_at": getattr(agent, "started_at", None),
                "tool": str(getattr(agent, "tool", "") or ""),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        base_identity = "call:" + hashlib.sha256(
            identity_payload.encode("utf-8", errors="replace")
        ).hexdigest()
        occurrence = seen.get(base_identity, 0)
        seen[base_identity] = occurrence + 1
        page_identity = (
            base_identity
            if occurrence == 0
            else f"{base_identity}:{occurrence}"
        )
        identified.append((agent, page_identity))
    return identified


def _execution_content_block(raw: object):
    """Normalize live frozen blocks and JSON-restored block dictionaries."""
    if not isinstance(raw, Mapping):
        if hasattr(raw, "kind") and hasattr(raw, "block_id"):
            return raw
        return None
    payload = dict(raw)
    kind = str(payload.pop("kind", "text") or "text")
    return ContentBlock(kind=kind, **payload)


_EXECUTION_TERMINAL_STATUSES: dict[AgentStatus, TerminalStatus] = {
    AgentStatus.DONE: "completed",
    AgentStatus.CACHED: "completed",
    AgentStatus.FAILED: "failed",
    AgentStatus.CANCELLED: "cancelled",
}


def _execution_terminal(agent: AgentProgress) -> TerminalStatus:
    """Map an agent progress status to the card state's terminal status."""
    return _EXECUTION_TERMINAL_STATUSES.get(agent.status, "running")


def _execution_header_template(agent: AgentProgress) -> str:
    """Return the Feishu header color template for a direct-call agent card."""
    return {
        AgentStatus.DONE: "green",
        AgentStatus.CACHED: "turquoise",
        AgentStatus.FAILED: "red",
        AgentStatus.CANCELLED: "grey",
    }.get(agent.status, "blue")


def _terminalize_execution_blocks(
    blocks: tuple,
    agent: AgentProgress,
) -> tuple:
    """Ensure a terminal direct call has no misleading active block chrome."""
    if _execution_terminal(agent) == "running":
        return blocks
    normalized = []
    for block in blocks:
        if getattr(block, "status", None) != "active":
            normalized.append(block)
            continue
        changes: dict[str, Any] = {"status": "completed"}
        if getattr(block, "kind", None) == "tool_call":
            changes["status"] = (
                "failed" if agent.status == AgentStatus.FAILED else "completed"
            )
            changes["is_latest_active"] = False
        normalized.append(dataclasses.replace(block, **changes))
    return tuple(normalized)


_EXECUTION_DISPLAY_TEXT_FIELDS = frozenset(
    {
        "alt",
        "content",
        "phase_name",
        "source_label",
        "source_sequence",
        "status_emoji",
        "task_name",
        "tool_input",
        "tool_name",
        "tool_output",
        "tool_summary",
    }
)


def _prepare_execution_blocks_for_wire(blocks: tuple) -> tuple:
    """Apply final-wire text transforms before execution-card pagination.

    Workflow cards pass through the delivery audit/table guard after the
    ordinary card renderer has already paginated them.  Audit replacement and
    Markdown-table neutralization can grow a page, which previously made that
    final guard truncate otherwise complete direct-call streams.  Mirror those
    display-text transforms on the immutable blocks first so pagination sees
    the bytes that will actually be sent.
    """
    prepared: list[Any] = []
    for block in blocks:
        changes: dict[str, str] = {}
        for field in dataclasses.fields(block):
            if field.name not in _EXECUTION_DISPLAY_TEXT_FIELDS:
                continue
            value = getattr(block, field.name)
            if not isinstance(value, str):
                continue
            safe_value = sanitize_card_text_for_audit(
                sanitize_markdown_image_references(
                    neutralize_feishu_rich_text_controls(
                        _utf8_safe_text(value)
                    )
                )
            )
            if safe_value != value:
                changes[field.name] = safe_value
        prepared.append(
            dataclasses.replace(block, **changes) if changes else block
        )

    table_count = sum(
        count_markdown_table_blocks(value)
        for block in prepared
        for field in dataclasses.fields(block)
        if field.name in _EXECUTION_DISPLAY_TEXT_FIELDS
        and isinstance((value := getattr(block, field.name)), str)
    )
    if table_count <= FEISHU_CARD_TABLE_LIMIT:
        return tuple(prepared)

    warning_pending = True
    normalized: list[Any] = []
    for block in prepared:
        changes = {}
        for field in dataclasses.fields(block):
            if field.name not in _EXECUTION_DISPLAY_TEXT_FIELDS:
                continue
            value = getattr(block, field.name)
            if not isinstance(value, str):
                continue
            contains_table = count_markdown_table_blocks(value) > 0
            if not contains_table:
                continue
            changes[field.name] = normalize_markdown_tables_for_card(
                value,
                table_limit=0,
                include_warning=warning_pending,
            )
            warning_pending = False
        normalized.append(
            dataclasses.replace(block, **changes) if changes else block
        )
    return tuple(normalized)


_AGENT_CARD_CACHE_MAX = 512
_AGENT_CARD_CACHE_LOCK = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
_AGENT_CARD_CACHE: OrderedDict[tuple[Any, ...], str] = OrderedDict()


def clear_agent_execution_card_cache() -> None:
    """Clear the bounded LRU cache of rendered workflow agent execution cards."""
    with _AGENT_CARD_CACHE_LOCK:
        _AGENT_CARD_CACHE.clear()


def _block_cache_token(block: Any) -> Any:
    """Return a hashable cache token for an execution ContentBlock."""
    try:
        return (type(block).__name__, hash(block), block)
    except TypeError:
        return (
            type(block).__name__,
            json.dumps(
                dataclasses.asdict(block),
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            ),
        )


def _render_agent_execution_cards(project: WorkflowProject) -> list[dict[str, Any]]:
    """Render one ordinary-format, losslessly paginated stream per direct call."""
    cards: list[dict[str, Any]] = []
    project_name = _safe_dynamic_text(project.name or "Workflow")
    render_card_fn = _resolve_renderer_attr("render_card", _default_render_card)
    for agent, page_identity in _identified_direct_agents(project):
        raw_blocks = list(getattr(agent, "execution_blocks", ()) or ())
        blocks = tuple(
            block
            for raw in raw_blocks
            if (block := _execution_content_block(raw)) is not None
        )
        if not blocks:
            continue

        display_index = max(1, int(getattr(agent, "call_index", 0) or 0) + 1)
        agent_id = _safe_dynamic_text(
            getattr(agent, "agent_id", None) or "unbound",
            one_line=True,
        )
        label = _safe_dynamic_text(agent.label or "agent", one_line=True)
        role = _safe_dynamic_text(getattr(agent, "role", None) or "", one_line=True)
        terminal = _execution_terminal(agent)
        header_template = _execution_header_template(agent)
        mode_name = _safe_dynamic_text(agent.tool or "Agent", one_line=True)
        tool_name = (
            _safe_dynamic_text(agent.tool, one_line=True)
            if agent.tool
            else None
        )
        _, clean_model, profile, effort = _binding_model_parts(
            agent,
            default_tool="Agent",
            default_model="",
        )
        if clean_model or profile or effort:
            base_model = clean_model or "default"
            suffix_parts = [part for part in (profile, effort) if part]
            raw_display_model = (
                f"{base_model} ({' / '.join(suffix_parts)})"
                if suffix_parts
                else base_model
            )
            model_name: str | None = _safe_dynamic_text(raw_display_model, one_line=True)
        else:
            model_name = None
        unit_label = (
            f"{agent_id} · {role} · {label}"
            if role
            else f"{agent_id} · {label}"
        )
        elapsed = max(0.0, float(agent.duration_s or 0.0))
        if elapsed <= 0 and agent.started_at:
            elapsed = max(
                0.0,
                float((agent.finished_at or time.time()) - agent.started_at),
            )
        elapsed_key = round(elapsed, 3) if terminal != "running" else int(elapsed)
        cache_key = (
            display_index,
            agent_id,
            label,
            role,
            profile or "",
            effort or "",
            terminal,
            header_template,
            agent.status == AgentStatus.FAILED,
            project_name,
            mode_name,
            tool_name,
            model_name,
            agent.started_at,
            elapsed_key,
            _WORKFLOW_FRAGMENT_MAX_BYTES,
            _WORKFLOW_FRAGMENT_MAX_NODES,
            tuple(_block_cache_token(block) for block in blocks),
        )

        with _AGENT_CARD_CACHE_LOCK:
            cached_json = _AGENT_CARD_CACHE.get(cache_key)
            if cached_json is not None:
                _AGENT_CARD_CACHE.move_to_end(cache_key)

        if cached_json is not None:
            page_dicts: list[dict[str, Any]] = json.loads(cached_json)
        else:
            wire_blocks = _terminalize_execution_blocks(blocks, agent)
            wire_blocks = _prepare_execution_blocks_for_wire(wire_blocks)
            state = CardState(
                blocks=wire_blocks,
                terminal=terminal,
                header=HeaderState(
                    title=f"#{display_index} · {unit_label}",
                    template=header_template,
                ),
                metadata=CardMetadata(
                    project_name=project_name,
                    mode_name=mode_name,
                    tool_name=tool_name,
                    model_name=model_name,
                    unit_id=str(display_index),
                    unit_kind="workflow_agent",
                    unit_label=unit_label,
                    card_sequence=display_index,
                    session_started_at=agent.started_at,
                    programming_text_sections=True,
                    full_execution_blocks=True,
                ),
                runtime_stats=RuntimeStats(elapsed_seconds=elapsed),
                version=len(wire_blocks),
                structural_version=len(wire_blocks),
            )
            rendered_pages = render_card_fn(
                state,
                RenderBudget(
                    byte_budget=_WORKFLOW_FRAGMENT_MAX_BYTES,
                    node_budget=_WORKFLOW_FRAGMENT_MAX_NODES,
                    engine_cmd="/wf",
                ),
            )
            page_dicts = [rendered.to_feishu_json() for rendered in rendered_pages]
            serialized = json.dumps(
                page_dicts,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            with _AGENT_CARD_CACHE_LOCK:
                _AGENT_CARD_CACHE[cache_key] = serialized
                _AGENT_CARD_CACHE.move_to_end(cache_key)
                while len(_AGENT_CARD_CACHE) > _AGENT_CARD_CACHE_MAX:
                    _AGENT_CARD_CACHE.popitem(last=False)

        for local_page_index, card in enumerate(page_dicts):
            _set_workflow_page_key(
                card,
                kind="agent",
                page_identity=page_identity,
                local_page_index=local_page_index,
            )
            body = card.get("body") or {}
            elements = body.get("elements") or []
            _card_text_for_agent_output(elements)
            cards.append(card)
    return cards


# ---------------------------------------------------------------------------
# Card size enforcement
# ---------------------------------------------------------------------------


def _paginate_progress_cards(
    header: dict[str, Any],
    elements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Split progress elements into cards without deleting scheduling rows."""

    def _card(page_elements: list[dict[str, Any]], page: int = 0, total: int = 0) -> dict[str, Any]:
        page_header = {
            **header,
            "title": dict(header.get("title") or {}),
        }
        if total > 1:
            title = str(page_header["title"].get("content") or "Workflow")
            page_header["title"]["content"] = f"{title} · {page + 1}/{total}"
        return {"header": page_header, "elements": page_elements}

    pages: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for element in elements:
        candidate = [*current, element]
        encoded = json.dumps(_card(candidate, 999, 999), ensure_ascii=False).encode(
            "utf-8", errors="surrogatepass"
        )
        if (
            len(encoded) <= _WORKFLOW_FRAGMENT_MAX_BYTES
            and count_tagged_nodes(_card(candidate, 999, 999))
            <= _WORKFLOW_FRAGMENT_MAX_NODES
        ):
            current = candidate
            continue
        if current:
            pages.append(current)
            current = []
        single_card = _card([element], 999, 999)
        single = json.dumps(single_card, ensure_ascii=False).encode(
            "utf-8", errors="surrogatepass"
        )
        if (
            len(single) > _WORKFLOW_FRAGMENT_MAX_BYTES
            or count_tagged_nodes(single_card) > _WORKFLOW_FRAGMENT_MAX_NODES
        ):
            raise ValueError("Workflow progress element exceeds Feishu card limit")
        current = [element]
    if current or not pages:
        pages.append(current)
    total = len(pages)
    return [_card(page, index, total) for index, page in enumerate(pages)]
