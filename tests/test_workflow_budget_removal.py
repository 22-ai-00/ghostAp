"""Backward-compatible workflow payload fields retained after budget removal."""

from src.card.events.payloads import (
    WorkflowConfirmPayloadOptional,
    WorkflowProgressPayload,
)


def test_deprecated_budget_payload_fields_remain_readable() -> None:
    assert {"budget_consumed", "budget_remaining"} <= set(
        WorkflowProgressPayload.__annotations__
    )
    assert "budget_total" in WorkflowConfirmPayloadOptional.__annotations__
