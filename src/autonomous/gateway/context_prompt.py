"""Deterministic rendering of an already-budgeted employee Context snapshot."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field

from ..context.models import AssembledContext, ContextMessage
from ..ingress.targeted_task import (
    TARGETED_TASK_DIGEST_VERSION,
    TARGETED_TASK_INPUT_KIND,
    targeted_group_task_digest,
)

_RENDER_CONTRACT = "ghostap.employee-context-prompt.v3:canonical-json"
RENDER_CONTRACT_DIGEST = hashlib.sha256(_RENDER_CONTRACT.encode()).hexdigest()


class EmployeePromptRenderError(ValueError):
    """The authenticated Context cannot be rendered within its frozen budget."""


@dataclass(frozen=True, slots=True)
class RenderedEmployeePrompt:
    prompt: str = field(repr=False)
    render_contract_digest: str
    context_snapshot_hash: str
    context_watermark_digest: str


@dataclass(frozen=True, slots=True)
class UntrustedCurrentMessageOverride:
    """One authenticated, derived view of the untrusted current message."""

    message_id: str
    text: str = field(repr=False)
    input_kind: str
    input_digest: str
    payload_digest: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.message_id, str)
            or not self.message_id.startswith("om_")
        ):
            raise EmployeePromptRenderError("override message_id is invalid")
        if not isinstance(self.text, str) or not self.text:
            raise EmployeePromptRenderError("override text is required")
        if self.input_kind != TARGETED_TASK_INPUT_KIND:
            raise EmployeePromptRenderError("override input kind is invalid")
        if self.input_digest != targeted_group_task_digest(self.text):
            raise EmployeePromptRenderError(
                "override input digest does not match text"
            )
        if re.fullmatch(r"[0-9a-f]{64}", self.payload_digest) is None:
            raise EmployeePromptRenderError("override payload digest is invalid")


def render_employee_context(
    snapshot: AssembledContext,
    *,
    system_instruction: str = "",
    constraints_digest: str = "",
    current_message_override: UntrustedCurrentMessageOverride | None = None,
) -> RenderedEmployeePrompt:
    """Render only retained fields, preserving the assembler's budget decision."""

    if not isinstance(snapshot, AssembledContext):
        raise TypeError("snapshot must be AssembledContext")
    if not snapshot.snapshot_hash:
        raise EmployeePromptRenderError(
            "context snapshot must carry its assembler hash"
        )
    if current_message_override is not None:
        current = tuple(
            message
            for message in snapshot.thread_messages
            if message.is_current
        )
        if (
            len(current) != 1
            or current[0].message_id != current_message_override.message_id
        ):
            raise EmployeePromptRenderError(
                "override current message is not bound to Context"
            )
        if len(current_message_override.text) > len(current[0].text):
            raise EmployeePromptRenderError(
                "override current message exceeds assembled budget"
            )
    untrusted_payload = {
        "thread": _message_payload(
            snapshot.thread_messages,
            current_message_override=current_message_override,
        ),
        "l1_memory": snapshot.l1_summary,
        "recent_group": _message_payload(
            snapshot.group_messages,
            current_message_override=current_message_override,
        ),
        "l2_group_memory": snapshot.l2_summary,
    }
    if not any(untrusted_payload.values()):
        raise EmployeePromptRenderError("already-budgeted context is empty")
    untrusted_json = json.dumps(
        untrusted_payload,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    untrusted_prompt = f"## UNTRUSTED_CONTEXT_JSON\n{untrusted_json}"
    override_contract = (
        ""
        if current_message_override is None
        else f"targeted-current-message-overlay:{TARGETED_TASK_DIGEST_VERSION}"
    )
    render_contract_digest = (
        RENDER_CONTRACT_DIGEST
        if not override_contract
        else hashlib.sha256(
            f"{_RENDER_CONTRACT}\0{override_contract}".encode()
        ).hexdigest()
    )
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
            (
                f"{_RENDER_CONTRACT}\0{trusted_payload}"
                f"\0{override_contract}"
            ).encode()
        ).hexdigest()
    else:
        prompt = untrusted_prompt
    raw_context_chars = (
        sum(
            len(item["text"])
            for item in untrusted_payload["thread"]
        )
        + sum(
            len(item["text"])
            for item in untrusted_payload["recent_group"]
        )
        + len(snapshot.l1_summary)
        + len(snapshot.l2_summary)
    )
    reserved_chars = len(prompt) - raw_context_chars
    reserved_tokens = math.ceil(reserved_chars * snapshot.tokens_per_char)
    if reserved_tokens > snapshot.system_prompt_tokens_reserved:
        raise EmployeePromptRenderError(
            "employee prompt envelope exceeds reserved budget"
        )
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
        context_snapshot_hash=(
            snapshot.snapshot_hash
            if current_message_override is None
            else hashlib.sha256(
                (
                    f"ghostap.effective-context.v1\0{snapshot.snapshot_hash}"
                    f"\0{current_message_override.message_id}"
                    f"\0{current_message_override.input_kind}"
                    f"\0{current_message_override.input_digest}"
                    f"\0{current_message_override.payload_digest}"
                    f"\0{hashlib.sha256(untrusted_json.encode()).hexdigest()}"
                ).encode()
            ).hexdigest()
        ),
        context_watermark_digest=(
            snapshot.watermark.revision_digest
            if snapshot.watermark is not None
            else hashlib.sha256(watermark_bytes).hexdigest()
        ),
    )


def _message_payload(
    messages: tuple[ContextMessage, ...],
    *,
    current_message_override: UntrustedCurrentMessageOverride | None = None,
) -> list[dict[str, str]]:
    return [
        {
            "message_id": message.message_id,
            "sender_id": message.sender_id,
            "text": (
                current_message_override.text
                if current_message_override is not None
                and message.is_current
                and message.message_id == current_message_override.message_id
                else message.text
            ),
        }
        for message in messages
    ]


__all__ = [
    "RENDER_CONTRACT_DIGEST",
    "EmployeePromptRenderError",
    "RenderedEmployeePrompt",
    "UntrustedCurrentMessageOverride",
    "render_employee_context",
]
