"""Tests for card event types and conversion."""

import pytest

from src.acp.models import (
    ACPEvent,
    ACPEventType,
    ACPImageInfo,
    PlanEntryInfo,
    PlanInfo,
    ToolCallInfo,
)
from src.card.events import CardEvent, CardEventType


class TestCardEventCreation:
    def test_started_factory(self):
        e = CardEvent.started()
        assert e.type == CardEventType.STARTED
        assert e.payload == {}

    @pytest.mark.parametrize("factory_name", ["image_added", "image_failed"])
    def test_image_alt_sanitizes_surrogates_and_control_characters(
        self,
        factory_name: str,
    ):
        factory = getattr(CardEvent, factory_name)
        if factory_name == "image_added":
            event = factory(
                "sha256:unsafe-name",
                "img_safe",
                "screen\udcff\x00\nshot.png",
            )
        else:
            event = factory(
                "sha256:unsafe-name",
                "screen\udcff\x00\nshot.png",
            )

        alt = event.payload["alt"]
        alt.encode("utf-8")
        assert "\x00" not in alt
        assert "\n" not in alt
        assert "\udcff" not in alt
        assert alt == "screen� shot.png"

    @pytest.mark.parametrize(
        "duration",
        [-1.0, float("nan"), float("inf"), True],
    )
    def test_terminal_factory_rejects_invalid_authoritative_duration(
        self,
        duration,
    ):
        with pytest.raises(ValueError, match="finite non-negative"):
            CardEvent.completed(duration_seconds=duration)

        with pytest.raises(ValueError, match="finite non-negative"):
            CardEvent.failed("boom", duration_seconds=duration)

    def test_terminal_factory_accepts_zero_authoritative_duration(self):
        assert CardEvent.completed(
            duration_seconds=0,
        ).payload["duration_seconds"] == 0.0
        assert CardEvent.failed(
            "boom",
            duration_seconds=0,
        ).payload["duration_seconds"] == 0.0

    def test_failed_factory(self):
        e = CardEvent.failed("oops")
        assert e.type == CardEventType.FAILED
        assert e.payload["error"] == "oops"

    def test_failed_no_arg_fallback(self):
        e = CardEvent.failed()
        assert e.payload["error"] == ""

    def test_blocked_factory(self):
        e = CardEvent.blocked("quota exceeded")
        assert e.type == CardEventType.BLOCKED
        assert e.payload["reason"] == "quota exceeded"

    def test_blocked_factory_empty_reason(self):
        e = CardEvent.blocked()
        assert e.type == CardEventType.BLOCKED
        assert e.payload.get("reason", "") == ""

    def test_review_result_updated_factory(self):
        e = CardEvent.review_result_updated(
            1,
            [{"role_id": "tester", "title": "测试工程师", "suggestions": ["补测试"]}],
        )

        assert e.type == CardEventType.REVIEW_RESULT_UPDATED
        assert e.payload["cycle_num"] == 1
        assert e.payload["roles"][0]["title"] == "测试工程师"

    def test_spec_plan_updated_factory(self):
        e = CardEvent.spec_plan_updated(
            1,
            {
                "architecture": "复用 CardSession 结构化事件，不恢复 raw JSON 流。",
                "steps": ["新增 PLAN 展示事件", "渲染方案规划面板"],
                "file_changes": ["src/card/events/factories.py"],
                "test_plan": ["新增卡片渲染回归"],
                "risks": [],
            },
        )

        assert e.type == CardEventType.SPEC_PLAN_UPDATED
        assert e.payload["cycle_num"] == 1
        assert e.payload["plan"]["steps"] == ["新增 PLAN 展示事件", "渲染方案规划面板"]

    def test_spec_tasks_updated_factory_preserves_full_descriptions(self):
        full_description = "任务 1 需要完整显示从事件层传来的描述，避免后续 build 阶段提到任务 1 时用户不知道对应内容"
        e = CardEvent.spec_tasks_updated(
            1,
            [
                {"task_id": 1, "description": full_description, "dependencies": []},
                {"task_id": 3, "description": "任务 3 也必须保留完整说明", "dependencies": [1]},
            ],
        )

        assert e.type == CardEventType.SPEC_TASKS_UPDATED
        assert e.payload["tasks"][0]["description"] == full_description
        assert e.payload["tasks"][1]["dependencies"] == [1]

    def test_text_delta_factory(self):
        e = CardEvent.text_delta("b1", "hello")
        assert e.type == CardEventType.TEXT_DELTA
        assert e.payload == {"block_id": "b1", "text": "hello"}

    def test_tool_started_factory(self):
        e = CardEvent.tool_started("t1", "bash", "ls -la")
        assert e.payload == {"block_id": "t1", "tool_name": "bash", "tool_input": "ls -la"}

    def test_card_split_factory(self):
        from src.card.events.payloads import CardSplitPayload

        e = CardEvent.card_split(reason="task_done", hint="接续 task 3")

        assert e.type == CardEventType.CARD_SPLIT
        assert e.type.value == "card_split"
        payload: CardSplitPayload = e.payload
        assert payload["reason"] == "task_done"
        assert payload["hint"] == "接续 task 3"

    def test_frozen(self):
        e = CardEvent.started()
        with pytest.raises(Exception):
            e.type = CardEventType.COMPLETED


class TestFromACP:
    def test_text_chunk(self):
        acp = ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="hi")
        ce = CardEvent.from_acp(acp)
        assert ce.type == CardEventType.TEXT_DELTA
        assert ce.payload["text"] == "hi"

    def test_thought_chunk(self):
        acp = ACPEvent(event_type=ACPEventType.THOUGHT_CHUNK, text="hmm")
        ce = CardEvent.from_acp(acp)
        assert ce.type == CardEventType.REASONING_DELTA
        assert ce.payload["text"] == "hmm"

    def test_image_chunk_without_media_bridge_has_visible_fallback(self):
        image = ACPImageInfo(
            image_id="sha256:abc",
            mime_type="image/png",
            data="aW1hZ2U=",
            name="截图",
        )

        ce = CardEvent.from_acp(
            ACPEvent(event_type=ACPEventType.IMAGE_CHUNK, image=image)
        )

        assert ce.type == CardEventType.IMAGE_FAILED
        assert ce.payload == {"image_id": "sha256:abc", "alt": "截图"}

    def test_tool_call_start(self):
        tc = ToolCallInfo(id="tc1", title="bash", kind="execute", status="in_progress", content="ls")
        acp = ACPEvent(event_type=ACPEventType.TOOL_CALL_START, tool_call=tc)
        ce = CardEvent.from_acp(acp)
        assert ce.type == CardEventType.TOOL_STARTED
        assert ce.payload["block_id"] == "tc1"
        assert ce.payload["tool_name"] == "bash"

    def test_tool_call_done(self):
        tc = ToolCallInfo(id="tc1", title="bash", kind="execute", status="completed", content="output")
        acp = ACPEvent(event_type=ACPEventType.TOOL_CALL_DONE, tool_call=tc)
        ce = CardEvent.from_acp(acp)
        assert ce.type == CardEventType.TOOL_DONE
        assert ce.payload["tool_output"] == "output"

    def test_tool_call_done_failed(self):
        tc = ToolCallInfo(id="tc1", title="bash", kind="execute", status="failed", content="err")
        acp = ACPEvent(event_type=ACPEventType.TOOL_CALL_DONE, tool_call=tc)
        ce = CardEvent.from_acp(acp)
        assert ce.type == CardEventType.TOOL_FAILED
        assert ce.payload["error"] == "err"

    def test_plan_update(self):
        plan = PlanInfo(entries=[
            PlanEntryInfo(content="Step 1", status="completed"),
            PlanEntryInfo(content="Step 2", status="in_progress"),
            PlanEntryInfo(content="Step 3", status="pending"),
        ])
        acp = ACPEvent(event_type=ACPEventType.PLAN_UPDATE, plan=plan)
        ce = CardEvent.from_acp(acp)
        assert ce.type == CardEventType.TASK_LIST_UPDATED
        assert ce.payload["current_task_id"] == "step_1"
        assert ce.payload["tasks"] == [
            {"task_id": "step_0", "name": "Step 1", "status": "completed"},
            {"task_id": "step_1", "name": "Step 2", "status": "in_progress"},
            {"task_id": "step_2", "name": "Step 3", "status": "pending"},
        ]

    def test_tool_call_update(self):
        tc = ToolCallInfo(id="tc1", title="bash", kind="execute", status="in_progress", content="partial output")
        acp = ACPEvent(event_type=ACPEventType.TOOL_CALL_UPDATE, tool_call=tc)
        ce = CardEvent.from_acp(acp)
        assert ce.type == CardEventType.TOOL_DELTA
        assert ce.payload["block_id"] == "tc1"
        assert ce.payload["content"] == "partial output"

    def test_unknown_event_type_fallback(self):
        """Unknown/unhandled event types should fall back to TEXT_DELTA."""
        acp = ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="fallback")
        # Patch the type to something the adapter doesn't explicitly handle
        acp.event_type = "totally_unknown_type"
        ce = CardEvent.from_acp(acp)
        assert ce.type == CardEventType.TEXT_DELTA


class TestCardEventCancelled:
    """Edge-case tests for CardEvent.cancelled()."""

    def test_cancelled_with_reason(self):
        e = CardEvent.cancelled(reason="ttl_expired")
        assert e.type == CardEventType.CANCELLED
        assert e.payload == {"reason": "ttl_expired"}

    def test_cancelled_without_reason(self):
        e = CardEvent.cancelled()
        assert e.type == CardEventType.CANCELLED
        assert e.payload == {}

    def test_cancelled_with_empty_string_reason(self):
        """Empty string reason should be treated as no reason (falsy)."""
        e = CardEvent.cancelled(reason="")
        assert e.type == CardEventType.CANCELLED
        # Empty string is falsy, so payload should be empty
        assert e.payload == {}

    def test_cancelled_with_none_reason(self):
        e = CardEvent.cancelled(reason=None)
        assert e.type == CardEventType.CANCELLED
        assert e.payload == {}
