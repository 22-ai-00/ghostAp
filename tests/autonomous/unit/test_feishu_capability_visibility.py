"""Unit tests for Feishu capability visibility and feature gating."""

from __future__ import annotations

from src.autonomous.feishu.provisioning import (
    FeishuCapabilities,
    ProbeStatus,
    unavailable,
)


def test_meeting_entry_hidden_when_gray_probe_fails() -> None:
    caps = FeishuCapabilities()
    caps.meeting_join = unavailable("ErrNotInGray")

    actions = caps.available_actions()
    assert "meeting" not in actions


def test_all_hidden_when_no_probes_run() -> None:
    caps = FeishuCapabilities()
    assert caps.available_actions() == []


def test_available_when_probe_succeeds() -> None:
    caps = FeishuCapabilities()
    caps.meeting_join.status = ProbeStatus.AVAILABLE
    caps.comment_thread.status = ProbeStatus.AVAILABLE

    actions = caps.available_actions()
    assert "meeting" in actions
    assert "comment" in actions
    assert "mirror" not in actions


def test_probe_all_with_no_fn_marks_unavailable() -> None:
    caps = FeishuCapabilities()
    caps.probe_all()

    assert caps.meeting_join.status is ProbeStatus.UNAVAILABLE
    assert caps.visible_employee.status is ProbeStatus.UNAVAILABLE


def test_probe_all_with_custom_fn() -> None:
    def fake_probe(capability: str) -> bool:
        return capability == "meeting_join"

    caps = FeishuCapabilities()
    caps.probe_all(fake_probe)

    assert caps.meeting_join.status is ProbeStatus.AVAILABLE
    assert caps.visible_employee.status is ProbeStatus.UNAVAILABLE


def test_probe_error_handling() -> None:
    def failing_probe(capability: str) -> bool:
        raise ConnectionError("network error")

    caps = FeishuCapabilities()
    caps.probe_all(failing_probe)

    assert caps.meeting_join.status is ProbeStatus.ERROR
    assert "network error" in caps.meeting_join.error
