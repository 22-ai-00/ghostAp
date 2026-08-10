"""Deterministic rendering of an already-budgeted employee Context snapshot."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

from ..context.models import AssembledContext, ContextMessage

_RENDER_CONTRACT = "ghostap.employee-context-prompt.v3:canonical-json"
RENDER_CONTRACT_DIGEST = hashlib.sha256(_RENDER_CONTRACT.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class RenderedEmployeePrompt:
    prompt: str
    render_contract_digest: str
    context_snapshot_hash: str
    context_watermark_digest: str


def render_employee_context(
    snapshot: AssembledContext,
    *,
    system_instruction: str = "",
    constraints_digest: str = "",
) -> RenderedEmployeePrompt:
    """Render only retained fields, preserving the assembler's budget decision."""

    if not isinstance(snapshot, AssembledContext):
        raise TypeError("snapshot must be AssembledContext")
    if not snapshot.snapshot_hash:
        raise ValueError("context snapshot must carry its assembler hash")
    untrusted_payload = {
        "thread": _message_payload(snapshot.thread_messages),
        "l1_memory": snapshot.l1_summary,
        "recent_group": _message_payload(snapshot.group_messages),
        "l2_group_memory": snapshot.l2_summary,
    }
    if not any(untrusted_payload.values()):
        raise ValueError("already-budgeted context is empty")
    untrusted_json = json.dumps(
        untrusted_payload,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    untrusted_prompt = f"## UNTRUSTED_CONTEXT_JSON\n{untrusted_json}"
    render_contract_digest = RENDER_CONTRACT_DIGEST
    if system_instruction:
        if not isinstance(system_instruction, str):
            raise TypeError("system_instruction must be text")
        trusted_payload = json.dumps(
            {
                "constraints_digest": constraints_digest,
                "persona": system_instruction,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        prompt = (
            "## TRUSTED_EMPLOYEE_SYSTEM_INSTRUCTION\n"
            f"{trusted_payload}\n\n"
            f"{untrusted_prompt}"
        )
        render_contract_digest = hashlib.sha256(
            f"{_RENDER_CONTRACT}\0{trusted_payload}".encode()
        ).hexdigest()
    else:
        prompt = untrusted_prompt
    raw_context_chars = (
        sum(len(message.text) for message in snapshot.thread_messages)
        + sum(len(message.text) for message in snapshot.group_messages)
        + len(snapshot.l1_summary)
        + len(snapshot.l2_summary)
    )
    reserved_chars = len(prompt) - raw_context_chars
    reserved_tokens = math.ceil(reserved_chars * snapshot.tokens_per_char)
    if reserved_tokens > snapshot.system_prompt_tokens_reserved:
        raise ValueError("employee prompt envelope exceeds reserved budget")
    watermark = None if snapshot.watermark is None else asdict(snapshot.watermark)
    watermark_bytes = json.dumps(
        watermark,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return RenderedEmployeePrompt(
        prompt=prompt,
        render_contract_digest=render_contract_digest,
        context_snapshot_hash=snapshot.snapshot_hash,
        context_watermark_digest=(
            snapshot.watermark.revision_digest
            if snapshot.watermark is not None
            else hashlib.sha256(watermark_bytes).hexdigest()
        ),
    )


def _message_payload(messages: tuple[ContextMessage, ...]) -> list[dict[str, str]]:
    return [
        {
            "message_id": message.message_id,
            "sender_id": message.sender_id,
            "text": message.text,
        }
        for message in messages
    ]


__all__ = [
    "RENDER_CONTRACT_DIGEST",
    "RenderedEmployeePrompt",
    "render_employee_context",
]
