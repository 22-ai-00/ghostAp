"""Tests for acp.client — GhostAPClient event handling."""

import asyncio
import base64
import logging
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from acp.schema import (
    AgentMessageChunk,
    ContentToolCallContent,
    EnvVariable,
    ImageContentBlock,
    PermissionOption,
    ResourceContentBlock,
    SessionInfoUpdate,
    TextContentBlock,
    ToolCallLocation,
    ToolCallProgress,
)

import src.acp.client as acp_client
from src.acp.client import ACPHistoryStore, GhostAPClient, _parse_plan, _parse_tool_call
from src.acp.models import ACPEvent, ACPGoalInfo, ACPSessionInfo
from src.acp.sync_adapter import resolve_agent_spec
from src.sandbox.executor import SandboxExecutor

_ONE_PIXEL_PNG = base64.b64encode(
    b"\x89PNG\r\n\x1a\n" + b"ghostap-image"
).decode("ascii")


def _session_info_update(codex: dict) -> SessionInfoUpdate:
    return SessionInfoUpdate.model_validate(
        {
            "sessionUpdate": "session_info_update",
            "_meta": {"codex": codex},
        }
    )


def test_codex_goal_session_info_is_control_plane_not_render_event(
    tmp_path: Path,
) -> None:
    events: list[ACPEvent] = []
    controls: list[tuple[str, ACPSessionInfo]] = []
    client = GhostAPClient(
        on_event=events.append,
        on_session_info=lambda sid, info: controls.append((sid, info)),
        root_dir=str(tmp_path),
    )
    update = _session_info_update(
        {
            "goal": {
                "objective": "finish the repository task",
                "status": "active",
                "tokenBudget": 12000,
                "timeUsedSeconds": 15,
                "createdAt": "2026-08-06T00:19:21Z",
                "controlMethod": "_codex/session/goal_control",
            },
            "threadStatus": {"type": "active"},
        }
    )

    asyncio.run(client.session_update("session-goal", update))

    assert events == []
    assert controls[0][0] == "session-goal"
    info = controls[0][1]
    assert info.goal_known is True
    assert info.goal is not None
    assert info.goal.status == "active"
    assert info.goal.token_budget == 12000
    assert info.goal.time_used_seconds == 15
    assert info.goal.control_method == "_codex/session/goal_control"
    assert info.thread_status_known is True
    assert info.thread_status == "active"


@pytest.mark.parametrize("status", ["paused", "completed"])
def test_codex_goal_session_info_accepts_terminal_status(
    tmp_path: Path,
    status: str,
) -> None:
    controls: list[tuple[str, ACPSessionInfo]] = []
    client = GhostAPClient(
        on_event=lambda _event: None,
        on_session_info=lambda sid, info: controls.append((sid, info)),
        root_dir=str(tmp_path),
    )

    asyncio.run(
        client.session_update(
            "session-goal",
            _session_info_update(
                {"goal": {"objective": "finish", "status": status}}
            ),
        )
    )

    assert controls[0][1].goal_known is True
    assert controls[0][1].goal is not None
    assert controls[0][1].goal.status == status


def test_codex_goal_session_info_null_is_known_clear(tmp_path: Path) -> None:
    controls: list[tuple[str, ACPSessionInfo]] = []
    client = GhostAPClient(
        on_event=lambda _event: None,
        on_session_info=lambda sid, info: controls.append((sid, info)),
        root_dir=str(tmp_path),
    )

    asyncio.run(
        client.session_update(
            "session-goal",
            _session_info_update({"goal": None}),
        )
    )

    assert controls[0][1].goal_known is True
    assert controls[0][1].goal is None


def test_codex_goal_session_info_accepts_finite_unix_created_at(
    tmp_path: Path,
) -> None:
    controls: list[tuple[str, ACPSessionInfo]] = []
    client = GhostAPClient(
        on_event=lambda _event: None,
        on_session_info=lambda sid, info: controls.append((sid, info)),
        root_dir=str(tmp_path),
    )

    asyncio.run(
        client.session_update(
            "session-goal",
            _session_info_update(
                {
                    "goal": {
                        "objective": "finish",
                        "status": "active",
                        "createdAt": 1785975561.25,
                    }
                }
            ),
        )
    )

    assert controls[0][1].goal is not None
    assert controls[0][1].goal.created_at == 1785975561.25


def test_codex_goal_session_info_missing_created_at_is_explicitly_absent(
    tmp_path: Path,
) -> None:
    controls: list[tuple[str, ACPSessionInfo]] = []
    client = GhostAPClient(
        on_event=lambda _event: None,
        on_session_info=lambda sid, info: controls.append((sid, info)),
        root_dir=str(tmp_path),
    )

    asyncio.run(
        client.session_update(
            "session-goal",
            _session_info_update(
                {"goal": {"objective": "finish", "status": "active"}}
            ),
        )
    )

    assert controls[0][1].goal is not None
    assert controls[0][1].goal.created_at is None


def test_unknown_goal_status_remains_visible_to_fail_closed_tracker(
    tmp_path: Path,
) -> None:
    controls: list[tuple[str, ACPSessionInfo]] = []
    client = GhostAPClient(
        on_event=lambda _event: None,
        on_session_info=lambda sid, info: controls.append((sid, info)),
        root_dir=str(tmp_path),
    )

    asyncio.run(
        client.session_update(
            "session-goal",
            _session_info_update(
                {
                    "goal": {"objective": "finish", "status": "running"},
                    "threadStatus": {"type": "idle"},
                }
            ),
        )
    )

    assert controls[0][1].goal is not None
    assert controls[0][1].goal.status == "running"
    assert controls[0][1].goal.activity_state is None


@pytest.mark.parametrize(
    ("status", "activity_state"),
    [
        ("active", True),
        ("paused", False),
        ("blocked", False),
        ("completed", False),
        ("running", None),
    ],
)
def test_goal_status_classification_is_explicitly_tristate(
    status: str,
    activity_state: bool | None,
) -> None:
    goal = ACPGoalInfo(objective="finish", status=status)

    assert goal.activity_state is activity_state
    assert goal.is_active is (activity_state is True)


def test_unknown_thread_status_does_not_emit_control_update(
    tmp_path: Path,
) -> None:
    controls: list[tuple[str, ACPSessionInfo]] = []
    client = GhostAPClient(
        on_event=lambda _event: None,
        on_session_info=lambda sid, info: controls.append((sid, info)),
        root_dir=str(tmp_path),
    )

    asyncio.run(
        client.session_update(
            "session-goal",
            _session_info_update({"threadStatus": {"type": "busy"}}),
        )
    )

    assert controls == []


@pytest.mark.parametrize(
    "payload",
    [
        {"title": "renamed session", "sessionUpdate": "session_info_update"},
        {
            "sessionUpdate": "session_info_update",
            "_meta": {"codex": {"goal": {"objective": "missing status"}}},
        },
        {
            "sessionUpdate": "session_info_update",
            "_meta": {
                "codex": {
                    "goal": {
                        "objective": "finish",
                        "status": "active",
                        "tokenBudget": True,
                    }
                }
            },
        },
        {
            "sessionUpdate": "session_info_update",
            "_meta": {
                "codex": {
                    "goal": {
                        "objective": "finish",
                        "status": "active",
                        "timeUsedSeconds": float("inf"),
                    }
                }
            },
        },
        {
            "sessionUpdate": "session_info_update",
            "_meta": {"vendor": {"goal": None}},
        },
    ],
)
def test_untrustworthy_goal_session_info_cannot_clear_stored_state(
    tmp_path: Path,
    payload: dict,
) -> None:
    controls: list[tuple[str, ACPSessionInfo]] = []
    client = GhostAPClient(
        on_event=lambda _event: None,
        on_session_info=lambda sid, info: controls.append((sid, info)),
        root_dir=str(tmp_path),
    )

    asyncio.run(
        client.session_update(
            "session-goal",
            SessionInfoUpdate.model_validate(payload),
        )
    )

    assert controls == [] or controls[0][1].goal_known is False


def test_agent_image_chunk_emits_typed_acp_image_event(tmp_path: Path):
    events: list[ACPEvent] = []
    client = GhostAPClient(on_event=events.append, root_dir=str(tmp_path))
    update = AgentMessageChunk(
        session_update="agent_message_chunk",
        content=ImageContentBlock(
            type="image",
            data=_ONE_PIXEL_PNG,
            mime_type="image/png",
        ),
    )

    asyncio.run(client.session_update("session-image", update))

    assert len(events) == 1
    assert events[0].event_type.value == "image_chunk"
    assert events[0].image is not None
    assert events[0].image.mime_type == "image/png"
    assert events[0].image.data == _ONE_PIXEL_PNG
    assert events[0].image.image_id.startswith("sha256:")


@pytest.mark.parametrize(
    ("payload", "mime_type"),
    [
        (b"plain text pretending to be a png", "image/png"),
        (b"\x89PNG\r\n", "image/png"),
        (b"\x89PNG\r\n\x1a\npayload", "image/jpeg"),
    ],
)
def test_agent_image_rejects_spoofed_or_truncated_raster_bytes(
    tmp_path: Path,
    payload: bytes,
    mime_type: str,
):
    events: list[ACPEvent] = []
    client = GhostAPClient(on_event=events.append, root_dir=str(tmp_path))
    update = AgentMessageChunk(
        session_update="agent_message_chunk",
        content=ImageContentBlock(
            type="image",
            data=base64.b64encode(payload).decode("ascii"),
            mime_type=mime_type,
        ),
    )

    asyncio.run(client.session_update("session-image", update))

    assert events == []


def test_tool_content_image_emits_before_tool_completion(tmp_path: Path):
    events: list[ACPEvent] = []
    client = GhostAPClient(on_event=events.append, root_dir=str(tmp_path))
    update = ToolCallProgress(
        session_update="tool_call_update",
        tool_call_id="imagegen-1",
        title="imagegen",
        kind="other",
        status="completed",
        content=[
            ContentToolCallContent(
                type="content",
                content=ImageContentBlock(
                    type="image",
                    data=_ONE_PIXEL_PNG,
                    mime_type="image/png",
                    uri="file:///workspace/generated.png",
                ),
            )
        ],
    )

    asyncio.run(client.session_update("session-image", update))

    assert [event.event_type.value for event in events] == [
        "image_chunk",
        "tool_call_done",
    ]
    assert events[0].image is not None
    assert events[0].image.name == "generated.png"


def test_subagent_tool_image_uses_stable_child_thread_source(tmp_path: Path):
    events: list[ACPEvent] = []
    client = GhostAPClient(on_event=events.append, root_dir=str(tmp_path))
    update = ToolCallProgress.model_validate(
        {
            "sessionUpdate": "tool_call_update",
            "toolCallId": "activity-call-private",
            "title": "Subagent image",
            "kind": "other",
            "status": "completed",
            "content": [
                {
                    "type": "content",
                    "content": {
                        "type": "image",
                        "data": _ONE_PIXEL_PNG,
                        "mimeType": "image/png",
                    },
                }
            ],
            "_meta": {
                "codex": {
                    "subagent": {
                        "threadId": "thread-stable-private",
                        "path": "/root/card-audit",
                        "activity": "interacted",
                    }
                }
            },
        }
    )

    asyncio.run(client.session_update("session-image", update))

    assert [event.source_id for event in events] == [
        "thread-stable-private",
        "thread-stable-private",
    ]
    assert [event.event_type.value for event in events] == [
        "image_chunk",
        "tool_call_done",
    ]


def test_completed_tool_image_location_emits_live_image_event(tmp_path: Path):
    events: list[ACPEvent] = []
    client = GhostAPClient(on_event=events.append, root_dir=str(tmp_path))
    snapshot = client.snapshot_local_images()
    image_path = tmp_path / "screenshots" / "desktop.png"
    image_path.parent.mkdir()
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nlocation")
    update = ToolCallProgress(
        session_update="tool_call_update",
        tool_call_id="screenshot-1",
        title="Take screenshot",
        kind="execute",
        status="completed",
        locations=[ToolCallLocation(path="screenshots/desktop.png")],
    )

    asyncio.run(client.session_update("session-image", update))

    assert [event.event_type.value for event in events] == [
        "image_chunk",
        "tool_call_done",
    ]
    assert events[0].image is not None
    assert events[0].image.source_uri == str(image_path)
    client.release_local_image_snapshot(snapshot)


def test_completed_tool_raw_output_image_path_emits_live_image_event(tmp_path: Path):
    events: list[ACPEvent] = []
    client = GhostAPClient(on_event=events.append, root_dir=str(tmp_path))
    snapshot = client.snapshot_local_images()
    image_path = tmp_path / "artifacts" / "mobile.webp"
    image_path.parent.mkdir()
    image_path.write_bytes(b"RIFF\x08\x00\x00\x00WEBPpayload")
    update = ToolCallProgress(
        session_update="tool_call_update",
        tool_call_id="browser-1",
        title="Browser screenshot",
        kind="execute",
        status="completed",
        raw_output={"output": "Screenshot saved to `artifacts/mobile.webp`"},
    )

    asyncio.run(client.session_update("session-image", update))

    assert [event.event_type.value for event in events] == [
        "image_chunk",
        "tool_call_done",
    ]
    assert events[0].image is not None
    assert events[0].image.name == "mobile.webp"
    client.release_local_image_snapshot(snapshot)


def test_completed_tool_does_not_publish_preexisting_referenced_private_image(
    tmp_path: Path,
):
    private = tmp_path / "private.png"
    private.write_bytes(b"\x89PNG\r\n\x1a\nprivate")
    events: list[ACPEvent] = []
    client = GhostAPClient(on_event=events.append, root_dir=str(tmp_path))
    snapshot = client.snapshot_local_images()
    update = ToolCallProgress(
        session_update="tool_call_update",
        tool_call_id="browser-private",
        title="Browser screenshot",
        kind="execute",
        status="completed",
        raw_output={"output": "Screenshot saved to `private.png`"},
    )

    asyncio.run(client.session_update("session-image", update))

    assert [event.event_type.value for event in events] == ["tool_call_done"]
    client.release_local_image_snapshot(snapshot)


def test_incomplete_prompt_baseline_fails_closed_for_local_path(
    tmp_path: Path,
):
    private = tmp_path / "private.png"
    private.write_bytes(b"\x89PNG\r\n\x1a\nprivate")
    snapshot = acp_client.snapshot_local_image_artifacts(str(tmp_path))
    snapshot.files.pop(str(private))
    snapshot.complete = False
    update = ToolCallProgress(
        session_update="tool_call_update",
        tool_call_id="incomplete-baseline",
        title="Browser screenshot",
        kind="execute",
        status="completed",
        raw_output={"output": "Screenshot saved to `private.png`"},
    )

    assert acp_client._tool_call_images(
        update,
        root_dir=str(tmp_path),
        image_snapshot=snapshot,
    ) == []
    acp_client.release_local_image_artifact_snapshot(snapshot)


def test_skipped_baseline_directory_cannot_publish_preexisting_local_image(
    tmp_path: Path,
):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    private = git_dir / "private.png"
    private.write_bytes(b"\x89PNG\r\n\x1a\nprivate")
    snapshot = acp_client.snapshot_local_image_artifacts(str(tmp_path))
    update = ToolCallProgress(
        session_update="tool_call_update",
        tool_call_id="skipped-baseline",
        title="Browser screenshot",
        kind="execute",
        status="completed",
        raw_output={"output": "Screenshot saved to `.git/private.png`"},
    )

    assert acp_client._tool_call_images(
        update,
        root_dir=str(tmp_path),
        image_snapshot=snapshot,
    ) == []
    acp_client.release_local_image_artifact_snapshot(snapshot)


def test_read_tool_location_does_not_publish_existing_input_image(tmp_path: Path):
    image_path = tmp_path / "reference.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nreference")
    events: list[ACPEvent] = []
    client = GhostAPClient(on_event=events.append, root_dir=str(tmp_path))
    update = ToolCallProgress(
        session_update="tool_call_update",
        tool_call_id="read-1",
        title="Read image",
        kind="read",
        status="completed",
        locations=[ToolCallLocation(path="reference.png")],
    )

    asyncio.run(client.session_update("session-image", update))

    assert [event.event_type.value for event in events] == ["tool_call_done"]


@pytest.mark.parametrize("content_kind", ["image", "resource", "text"])
def test_read_tool_content_does_not_publish_existing_input_image(
    tmp_path: Path,
    content_kind: str,
):
    image_path = tmp_path / "reference.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nreference")
    if content_kind == "image":
        content = ImageContentBlock(
            type="image",
            data=_ONE_PIXEL_PNG,
            mime_type="image/png",
            uri=str(image_path),
        )
    elif content_kind == "resource":
        content = ResourceContentBlock(
            type="resource_link",
            uri="reference.png",
            name="reference.png",
            mime_type="image/png",
        )
    else:
        content = TextContentBlock(
            type="text",
            text="Input image: `reference.png`",
        )
    events: list[ACPEvent] = []
    client = GhostAPClient(on_event=events.append, root_dir=str(tmp_path))
    update = ToolCallProgress(
        session_update="tool_call_update",
        tool_call_id=f"read-content-{content_kind}",
        title="Read image",
        kind="read",
        status="completed",
        content=[
            ContentToolCallContent(
                type="content",
                content=content,
            )
        ],
    )

    asyncio.run(client.session_update("session-image", update))

    assert [event.event_type.value for event in events] == ["tool_call_done"]


def test_local_image_resource_is_limited_to_acp_project_root(tmp_path: Path):
    events: list[ACPEvent] = []
    client = GhostAPClient(on_event=events.append, root_dir=str(tmp_path))
    snapshot = client.snapshot_local_images()
    image_path = tmp_path / "screenshots" / "page.png"
    image_path.parent.mkdir()
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nlocal")
    update = AgentMessageChunk(
        session_update="agent_message_chunk",
        content=ResourceContentBlock(
            type="resource_link",
            uri="screenshots/page.png",
            name="page.png",
            mime_type="image/png",
        ),
    )

    asyncio.run(client.session_update("session-image", update))

    assert len(events) == 1
    assert events[0].image is not None
    assert events[0].image.name == "page.png"
    assert events[0].image.source_uri == str(image_path)
    client.release_local_image_snapshot(snapshot)


def test_remote_image_resource_is_not_fetched(tmp_path: Path):
    events: list[ACPEvent] = []
    client = GhostAPClient(on_event=events.append, root_dir=str(tmp_path))
    update = AgentMessageChunk(
        session_update="agent_message_chunk",
        content=ResourceContentBlock(
            type="resource_link",
            uri="https://example.invalid/private.png",
            name="private.png",
            mime_type="image/png",
        ),
    )

    asyncio.run(client.session_update("session-image", update))

    assert events == []


def test_typed_local_resource_with_untracked_suffix_fails_closed(
    tmp_path: Path,
):
    private = tmp_path / "private.bin"
    private.write_bytes(b"\x89PNG\r\n\x1a\nprivate")
    events: list[ACPEvent] = []
    client = GhostAPClient(on_event=events.append, root_dir=str(tmp_path))
    snapshot = client.snapshot_local_images()
    update = AgentMessageChunk(
        session_update="agent_message_chunk",
        content=ResourceContentBlock(
            type="resource_link",
            uri="private.bin",
            name="private.bin",
            mime_type="image/png",
        ),
    )

    asyncio.run(client.session_update("session-image", update))

    assert events == []
    client.release_local_image_snapshot(snapshot)


def test_local_image_read_rejects_file_swapped_to_symlink_after_resolution(
    tmp_path: Path,
    monkeypatch,
):
    root = tmp_path / "root"
    root.mkdir()
    inside = root / "inside.png"
    outside = tmp_path / "outside.png"
    inside.write_bytes(b"\x89PNG\r\n\x1a\ninside")
    outside.write_bytes(b"\x89PNG\r\n\x1a\noutside")
    original_resolve = acp_client._safe_resolve_path

    def resolve_then_swap(root_dir: str, user_path: str):
        resolved = original_resolve(root_dir, user_path)
        resolved.unlink()
        resolved.symlink_to(outside)
        return resolved

    monkeypatch.setattr(
        acp_client,
        "_safe_resolve_path",
        resolve_then_swap,
    )

    image = acp_client._read_local_image_resource(
        root_dir=str(root),
        uri="inside.png",
        mime_type="image/png",
        name=None,
    )

    assert image is None


def test_image_scan_stops_iterating_at_entry_budget(
    tmp_path: Path,
    monkeypatch,
):
    inspected = 0

    class FakeEntry:
        name = "not-an-image"
        path = str(tmp_path / name)

        def is_dir(self, *, follow_symlinks=True):
            return False

        def is_file(self, *, follow_symlinks=True):
            return False

    class FakeScandir:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            nonlocal inspected
            for _ in range(acp_client._MAX_IMAGE_SCAN_ENTRIES + 500):
                inspected += 1
                yield FakeEntry()

    monkeypatch.setattr(acp_client.os, "scandir", lambda _path: FakeScandir())

    assert acp_client._iter_local_image_files(str(tmp_path)) == []
    assert inspected <= acp_client._MAX_IMAGE_SCAN_ENTRIES + 1


def test_repeated_changed_image_scans_share_one_prompt_budget(
    tmp_path: Path,
) -> None:
    snapshot = acp_client.snapshot_local_image_artifacts(str(tmp_path))
    paths = [tmp_path / f"continuation-{index}.png" for index in range(3)]
    for index, path in enumerate(paths):
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([index]))

    events: list[ACPEvent] = []
    budget = acp_client.LocalImageEmissionBudget(
        max_images=2,
        max_bytes=1024,
    )
    try:
        acp_client.emit_referenced_changed_local_image_events(
            str(tmp_path),
            snapshot,
            str(paths[0]),
            events.append,
            budget=budget,
            release_snapshot=False,
        )
        acp_client.emit_referenced_changed_local_image_events(
            str(tmp_path),
            snapshot,
            " ".join(str(path) for path in paths),
            events.append,
            budget=budget,
            release_snapshot=False,
        )
    finally:
        acp_client.release_local_image_artifact_snapshot(snapshot)

    assert [event.image.source_uri for event in events if event.image] == [
        str(paths[0]),
        str(paths[1]),
    ]
    assert budget.emitted_count == 2
    assert budget.emitted_bytes > 0


def test_overlapping_prompt_scans_do_not_cross_publish_images(
    tmp_path: Path,
):
    """Ambiguous same-root writes must not be attributed to either prompt."""
    first_snapshot = acp_client.snapshot_local_image_artifacts(str(tmp_path))
    second_snapshot = acp_client.snapshot_local_image_artifacts(str(tmp_path))
    secret = tmp_path / "other-chat-secret.png"
    secret.write_bytes(b"\x89PNG\r\n\x1a\nother-chat")
    update = ToolCallProgress(
        session_update="tool_call_update",
        tool_call_id="ambiguous",
        title="Take screenshot",
        kind="execute",
        status="completed",
        raw_output={"output": f"saved `{secret}`"},
    )

    assert acp_client._tool_call_images(
        update,
        root_dir=str(tmp_path),
        image_snapshot=first_snapshot,
    ) == []
    assert acp_client._tool_call_images(
        update,
        root_dir=str(tmp_path),
        image_snapshot=second_snapshot,
    ) == []
    acp_client.release_local_image_artifact_snapshot(first_snapshot)
    acp_client.release_local_image_artifact_snapshot(second_snapshot)

    isolated_snapshot = acp_client.snapshot_local_image_artifacts(str(tmp_path))
    own_image = tmp_path / "own-result.png"
    own_image.write_bytes(b"\x89PNG\r\n\x1a\nown-result")
    isolated_update = ToolCallProgress(
        session_update="tool_call_update",
        tool_call_id="isolated",
        title="Take screenshot",
        kind="execute",
        status="completed",
        raw_output={"output": f"saved `{own_image}`"},
    )

    images = acp_client._tool_call_images(
        isolated_update,
        root_dir=str(tmp_path),
        image_snapshot=isolated_snapshot,
    )
    acp_client.release_local_image_artifact_snapshot(isolated_snapshot)
    assert len(images) == 1
    assert images[0].source_uri == str(own_image)


def test_parent_and_child_root_snapshots_fail_closed_for_shared_image(
    tmp_path: Path,
):
    nested = tmp_path / "nested"
    nested.mkdir()
    parent_snapshot = acp_client.snapshot_local_image_artifacts(str(tmp_path))
    child_snapshot = acp_client.snapshot_local_image_artifacts(str(nested))
    shared = nested / "shared.png"
    shared.write_bytes(b"\x89PNG\r\n\x1a\nshared")
    update = ToolCallProgress(
        session_update="tool_call_update",
        tool_call_id="shared-root",
        title="Take screenshot",
        kind="execute",
        status="completed",
        raw_output={"output": f"saved `{shared}`"},
    )

    assert acp_client._tool_call_images(
        update,
        root_dir=str(tmp_path),
        image_snapshot=parent_snapshot,
    ) == []
    assert acp_client._tool_call_images(
        update,
        root_dir=str(nested),
        image_snapshot=child_snapshot,
    ) == []
    acp_client.release_local_image_artifact_snapshot(parent_snapshot)
    acp_client.release_local_image_artifact_snapshot(child_snapshot)


def test_local_surrogateescape_filename_is_sanitized_before_image_event(
    tmp_path: Path,
):
    events: list[ACPEvent] = []
    client = GhostAPClient(on_event=events.append, root_dir=str(tmp_path))
    snapshot = client.snapshot_local_images()
    raw_path = os.fsencode(tmp_path) + b"/screen_\xff.png"
    with open(raw_path, "wb") as image_file:
        image_file.write(b"\x89PNG\r\n\x1a\nsurrogate-name")
    display_path = os.fsdecode(raw_path)
    update = ToolCallProgress(
        session_update="tool_call_update",
        tool_call_id="surrogate-name",
        title="Take screenshot",
        kind="execute",
        status="completed",
        raw_output={"output": f"saved `{display_path}`"},
    )

    asyncio.run(client.session_update("session-image", update))

    assert [event.event_type.value for event in events] == [
        "image_chunk",
        "tool_call_done",
    ]
    assert events[0].image is not None
    events[0].image.name.encode("utf-8")
    assert "\udcff" not in events[0].image.name
    assert events[0].image.name == "screen_�.png"
    client.release_local_image_snapshot(snapshot)


def test_prompt_does_not_emit_unreported_new_image(tmp_path: Path):
    from types import SimpleNamespace

    from src.acp.session import ACPSession

    existing = tmp_path / "existing.png"
    existing.write_bytes(b"\x89PNG\r\n\x1a\nexisting")
    generated = tmp_path / "screenshots" / "final.png"

    class FakeConn:
        async def prompt(self, **_kwargs):
            generated.parent.mkdir()
            generated.write_bytes(b"\x89PNG\r\n\x1a\ngenerated")
            return SimpleNamespace(stop_reason="end_turn")

    events: list[ACPEvent] = []
    session = ACPSession(agent_cmd="test", agent_args=[], cwd=str(tmp_path))
    session._conn = FakeConn()
    session._session_id = "session-image"
    session._client = GhostAPClient(
        on_event=session._dispatch_event,
        root_dir=str(tmp_path),
    )

    result = asyncio.run(session.prompt("generate a screenshot", on_event=events.append))

    assert result.stop_reason == "end_turn"
    image_events = [event for event in events if event.event_type.value == "image_chunk"]
    assert image_events == []


def test_prompt_drains_late_image_update_even_after_text(tmp_path: Path):
    from types import SimpleNamespace

    from src.acp.models import ACPEventType, ACPImageInfo
    from src.acp.session import ACPSession

    image = ACPImageInfo(
        image_id="sha256:late",
        mime_type="image/png",
        data=_ONE_PIXEL_PNG,
        name="late.png",
    )
    session = ACPSession(agent_cmd="test", agent_args=[], cwd=str(tmp_path))

    class FakeConn:
        async def prompt(self, **_kwargs):
            session._dispatch_event(
                ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="done")
            )
            asyncio.get_running_loop().call_later(
                0.01,
                session._dispatch_event,
                ACPEvent(event_type=ACPEventType.IMAGE_CHUNK, image=image),
            )
            return SimpleNamespace(stop_reason="end_turn")

    events: list[ACPEvent] = []
    session._conn = FakeConn()
    session._session_id = "session-image"
    session._client = GhostAPClient(
        on_event=session._dispatch_event,
        root_dir=str(tmp_path),
    )

    asyncio.run(session.prompt("generate", on_event=events.append))

    assert [event.event_type for event in events] == [
        ACPEventType.TEXT_CHUNK,
        ACPEventType.IMAGE_CHUNK,
    ]


def test_prompt_preserves_empty_stop_reason_for_fail_closed_classification(
    tmp_path: Path,
):
    from types import SimpleNamespace

    from src.acp.outcome import PromptOutcome, classify_prompt_result
    from src.acp.session import ACPSession

    class FakeConn:
        async def prompt(self, **_kwargs):
            return SimpleNamespace(stop_reason="")

    session = ACPSession(agent_cmd="test", agent_args=[], cwd=str(tmp_path))
    session._conn = FakeConn()
    session._session_id = "session-empty-stop"

    result = asyncio.run(session.prompt("run"))
    assessment = classify_prompt_result(result)

    assert result.stop_reason == ""
    assert assessment.outcome is PromptOutcome.INCOMPLETE
    assert assessment.stop_reason == "missing_stop_reason"


def test_acp_manager_retries_start_failure(monkeypatch, caplog):
    from types import SimpleNamespace

    from src.acp import manager as mgr
    from tests.helpers import FakeSessionBase

    calls = {"start": 0}

    class FakeSession(FakeSessionBase):
        def start(self, startup_timeout: float = 60):
            calls["start"] += 1
            if calls["start"] < 3:
                raise TimeoutError("startup timeout")
            self.session_id = "s_ok"
            return self.session_id

    monkeypatch.setattr(mgr, "SyncACPSession", FakeSession)
    monkeypatch.setattr(
        mgr, "get_settings", lambda: SimpleNamespace(acp_startup_retries=3, acp_healthcheck_timeout=0.01)
    )

    caplog.set_level(logging.WARNING)
    # 默认路径仍然支持「少参数」构造
    m = mgr.ACPSessionManager("coco", session_timeout=999999)
    # The startup timeout is one absolute budget shared by all retries.  Give
    # this retry-behaviour test enough budget for its two intentional backoffs.
    s = m.start_session("chat1", cwd=".", startup_timeout=3.0)
    assert s.session_id == "s_ok"
    assert calls["start"] == 3

    # 启动失败日志应包含稳定字段（即便具体值为空）
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "Session start failed" in joined
    assert '"cmd"' in joined
    assert '"args"' in joined
    assert '"rc"' in joined
    assert '"stdout_snippet"' in joined
    assert '"stderr_snippet"' in joined


def test_acp_manager_ttadk_start_failure_no_coco_acp_fallback(monkeypatch, caplog):
    """TTADK 必须坚持 CLI 路径：启动失败时直接报错，不降级到 Coco ACP。"""
    from types import SimpleNamespace

    from src.acp import manager as mgr
    from src.acp.startup_utils import StartupOperationalError
    from tests.helpers import FakeSessionBase

    class FakeFailCLISession(FakeSessionBase):
        def describe_agent(self):
            return "tool=coco backend=cli cwd=."

        def start(self, startup_timeout: float = 60):
            raise StartupOperationalError("boom_cli")

        def is_server_healthy(self, healthcheck_timeout: float = 2.0) -> bool:
            return False

    # 若走到 ACP fallback，此断言会失败。
    monkeypatch.setattr(
        mgr, "SyncACPSession", lambda **kw: (_ for _ in ()).throw(AssertionError("unexpected_acp_fallback"))
    )
    monkeypatch.setattr("src.agent_session.SyncTTADKCLISession", FakeFailCLISession)
    monkeypatch.setattr(
        "src.ttadk.startup_common.precheck_ttadk_startup_model",
        lambda **kw: {"model": None, "validated": False, "tool": "coco", "input_model": ""},
    )
    monkeypatch.setattr(
        mgr,
        "get_settings",
        lambda: SimpleNamespace(acp_startup_retries=1, acp_healthcheck_timeout=0.01, ttadk_preheat_enabled=False),
    )

    caplog.set_level(logging.WARNING)
    m = mgr.ACPSessionManager("ttadk", session_timeout=999999)
    with pytest.raises(RuntimeError, match="启动 ttadk_coco CLI 失败"):
        m.start_session("chat1", cwd=".", startup_timeout=0.01, agent_type_override="ttadk_coco")

    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "Error while starting TTADK CLI" in joined


def test_supports_acp_serve_unsets_claudecode(monkeypatch):
    """ACP serve 探测不应继承 nested-session guard 环境变量。"""
    from types import SimpleNamespace

    from src.acp import sync_adapter as sa

    # lru_cache: ensure isolation
    try:
        sa._supports_acp_serve.cache_clear()
    except Exception:
        pass

    calls = {"env": None}

    def fake_run(cmd, capture_output, text, timeout, env=None):
        calls["env"] = env
        return SimpleNamespace(stdout="ACP Server", stderr="")

    monkeypatch.setattr(sa.subprocess, "run", fake_run)
    with monkeypatch.context() as m:
        m.setenv("CLAUDECODE", "1")
        assert sa._supports_acp_serve("claude") is True
        assert calls["env"] is not None
        assert "CLAUDECODE" not in calls["env"]


def test_acp_session_start_passes_env_without_claudecode(monkeypatch):
    """ACPSession 启动时应主动剔除 CLAUDECODE，避免 Claude nested-session 检测。"""
    from types import SimpleNamespace

    import src.acp.session as session_mod
    from src.acp.session import ACPSession

    calls = {"env": None}

    class FakeConn:
        async def initialize(self, protocol_version: int = 1):
            return None

        async def new_session(self, cwd: str):
            return SimpleNamespace(session_id="s_test")

    class FakeProc:
        returncode = None

    class FakeCtx:
        async def __aenter__(self):
            return FakeConn(), FakeProc()

        async def __aexit__(self, exc_type, exc, tb):
            return None

    def fake_spawn(to_client, command, *args, env=None, cwd=None, transport_kwargs=None, **kw):
        calls["env"] = env
        return FakeCtx()

    monkeypatch.setattr(session_mod, "spawn_agent_process", fake_spawn)
    monkeypatch.setattr(
        session_mod,
        "get_settings",
        lambda: SimpleNamespace(acp_permission_auto_approve=True, acp_stream_buffer_limit=0),
    )

    with monkeypatch.context() as m:
        m.setenv("CLAUDECODE", "1")
        s = ACPSession(agent_cmd="claude", agent_args=["acp", "serve"], cwd="/tmp")
        sid = asyncio.run(s.start())
        assert sid == "s_test"
        assert calls["env"] is not None
        assert "CLAUDECODE" not in calls["env"]


def test_acp_session_start_failure_has_fail_phase(monkeypatch):
    """ACPSession.start 失败时应抛 ACPStartupError 且携带 fail_phase（spawn/initialize/new_session）。"""
    from types import SimpleNamespace

    import src.acp.session as session_mod
    from src.acp.session import ACPSession, ACPStartupError

    class FakeProc:
        returncode = 7
        stdout = None
        stderr = None

    class FakeConn:
        async def initialize(self, protocol_version: int = 1):
            raise RuntimeError("init failed")

    class FakeCtx:
        async def __aenter__(self):
            return FakeConn(), FakeProc()

        async def __aexit__(self, exc_type, exc, tb):
            return None

    def fake_spawn(to_client, command, *args, env=None, cwd=None, transport_kwargs=None, **kw):
        return FakeCtx()

    monkeypatch.setattr(session_mod, "spawn_agent_process", fake_spawn)
    monkeypatch.setattr(
        session_mod,
        "get_settings",
        lambda: SimpleNamespace(acp_permission_auto_approve=True, acp_stream_buffer_limit=0),
    )

    s = ACPSession(agent_cmd="claude", agent_args=["acp", "serve"], cwd="/tmp")
    with pytest.raises(ACPStartupError) as ctx:
        asyncio.run(s.start())

    e = ctx.value
    assert getattr(e, "fail_phase", "") in ("initialize", "spawn", "new_session", "unknown")


def test_acp_health_check_uses_non_mutating_session_list_probe():
    """Health checks must not reload or otherwise mutate the active session."""
    from types import SimpleNamespace

    from src.acp.session import ACPSession

    calls: list[tuple[str, object]] = []

    class FakeConn:
        async def list_sessions(self, *, cwd=None):
            calls.append(("list_sessions", cwd))
            return SimpleNamespace(sessions=[])

        async def load_session(self, **_kwargs):
            raise AssertionError("health check must not reload the live session")

    session = ACPSession(agent_cmd="traex", agent_args=["acp", "serve"], cwd="/repo")
    session._proc = SimpleNamespace(returncode=None)
    session._conn = FakeConn()
    session._session_id = "s_live"

    assert asyncio.run(session.health_check(timeout=0.1)) is True
    assert calls == [("list_sessions", "/repo")]

    session._session_id = ""
    assert asyncio.run(session.health_check(timeout=0.1)) is False
    assert calls == [("list_sessions", "/repo")]


def test_acp_manager_unhealthy_session_is_cleaned(monkeypatch):
    import time as _time
    from types import SimpleNamespace

    from src.acp import manager as mgr

    class DeadSession:
        def __init__(self):
            self.session_id = "s_dead"
            # Idle > 30s to trigger health check path in get_session
            self.last_active = _time.time() - 60
            self.message_count = 0
            self.closed = False

        def is_server_running(self) -> bool:
            return False  # process is dead

        def to_snapshot(self):
            return {"session_id": self.session_id}

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        mgr, "get_settings", lambda: SimpleNamespace(acp_healthcheck_timeout=0.01, acp_startup_retries=1)
    )

    m = mgr.ACPSessionManager("coco", session_timeout=999999)
    dead = DeadSession()
    key = m._session_key("chat1")
    m._sessions[key] = dead

    assert m.get_session("chat1") is None
    assert dead.closed is True
    assert key not in m._sessions


def test_acp_manager_session_starter_success_is_not_overwritten(monkeypatch):
    """回归：session_starter 成功返回后不应被默认路径覆盖。"""
    from types import SimpleNamespace

    from src.acp import manager as mgr

    class _StarterSession:
        def __init__(self):
            self.session_id = "sid_from_starter"
            self.last_active = 123.0
            self.message_count = 7

        def describe_agent(self):
            return "starter"

        def load_session(self, session_id: str, timeout: float = 60):
            del timeout
            self.session_id = session_id

        def load_local_history(self, *args, **kwargs):
            return []

        def to_snapshot(self):
            return {"session_id": self.session_id}

        def close(self):
            return None

        def is_server_running(self) -> bool:
            return True

        def is_server_healthy(self, healthcheck_timeout: float = 2.0) -> bool:
            return True

    # If fallback path is entered, this fake will explode and fail the test.
    class _ShouldNotBeUsed:
        def __init__(self, *args, **kwargs):
            raise AssertionError("fallback SyncACPSession should not be used")

    monkeypatch.setattr(mgr, "SyncACPSession", _ShouldNotBeUsed)
    monkeypatch.setattr(
        mgr, "get_settings", lambda: SimpleNamespace(acp_healthcheck_timeout=0.01, acp_startup_retries=1)
    )

    def _starter(**kwargs):
        return (_StarterSession(), "sid_from_starter", {"attempts": []})

    m = mgr.ACPSessionManager("coco", session_starter=_starter)
    s = m.start_session("chat1", cwd=".", startup_timeout=0.01)
    assert s.session_id == "sid_from_starter"


class MockToolCallStart:
    """Mock ToolCallStart ACP schema object."""

    def __init__(
        self,
        tool_call_id="tc1",
        title="Read file",
        kind="read",
        status="in_progress",
        locations=None,
        raw_input=None,
        raw_output=None,
        field_meta=None,
    ):
        self.tool_call_id = tool_call_id
        self.title = title
        self.kind = kind
        self.status = status
        self.locations = locations or []
        self.raw_input = raw_input
        self.raw_output = raw_output
        self.field_meta = field_meta


class MockToolCallProgress:
    """Mock ToolCallProgress ACP schema object."""

    def __init__(
        self,
        tool_call_id="tc1",
        title="Read file",
        kind="read",
        status="completed",
        locations=None,
        raw_input=None,
        raw_output=None,
        field_meta=None,
    ):
        self.tool_call_id = tool_call_id
        self.title = title
        self.kind = kind
        self.status = status
        self.locations = locations or []
        self.raw_input = raw_input
        self.raw_output = raw_output
        self.field_meta = field_meta


class MockLocation:
    def __init__(self, path):
        self.path = path


class MockPlanEntry:
    def __init__(self, content, priority="medium", status="pending"):
        self.content = content
        self.priority = priority
        self.status = status


class TestParseToolCall:
    def test_basic(self):
        update = MockToolCallStart(tool_call_id="tc1", title="Read", kind="read", status="in_progress")
        tc = _parse_tool_call(update)
        assert tc.id == "tc1"
        assert tc.title == "Read"
        assert tc.kind == "read"
        assert tc.status == "in_progress"
        assert tc.locations == []

    def test_with_locations(self):
        update = MockToolCallStart(
            locations=[MockLocation("/a.py"), MockLocation("/b.py")],
        )
        tc = _parse_tool_call(update)
        assert tc.locations == ["/a.py", "/b.py"]

    def test_none_title(self):
        update = MockToolCallStart(title=None)
        tc = _parse_tool_call(update)
        assert tc.title == ""

    def test_none_kind(self):
        update = MockToolCallStart(kind=None)
        tc = _parse_tool_call(update)
        assert tc.kind == "other"

    def test_agent_tool_keeps_task_description_for_task_cards(self):
        update = MockToolCallStart(
            title="Agent",
            kind="other",
            status="in_progress",
            raw_input={
                "description": "实现后端接口",
                "prompt": "请实现 `/api/tasks` 接口并补测试",
                "subagent_type": "Explore",
            },
        )
        tc = _parse_tool_call(update)
        assert "实现后端接口" in tc.content

    def test_codex_subagent_activity_metadata_is_normalized(self):
        update = MockToolCallStart(
            tool_call_id="activity-call-private",
            title="Start subagent card-audit",
            kind="other",
            status="in_progress",
            raw_input={
                "agentThreadId": "thread-private",
                "agentPath": "/root/card-audit",
                "activityKind": "started",
            },
            field_meta={
                "codex": {
                    "subagent": {
                        "threadId": "thread-private",
                        "path": "/root/card-audit",
                        "activity": "started",
                    }
                }
            },
        )

        tc = _parse_tool_call(update)

        assert tc.subagent_source_id == "thread-private"
        assert tc.subagent_path == "/root/card-audit"
        assert tc.subagent_activity == "started"

    def test_codex_collaboration_states_are_normalized_by_thread(self):
        update = MockToolCallStart(
            tool_call_id="collaboration-call-private",
            title="wait_agent",
            kind="other",
            status="completed",
            raw_input={
                "prompt": "等待卡片审计",
                "receiverThreadIds": ["thread-a", "thread-b"],
                "agentsStates": {
                    "thread-a": {"status": "running", "message": "正在核查普通卡"},
                    "thread-b": {"status": "completed", "message": "已核查 Deep 卡"},
                },
                "model": "gpt-test",
                "reasoningEffort": "high",
            },
            raw_output={"result": "collaboration snapshot"},
            field_meta={
                "codex": {
                    "collaboration": {
                        "tool": "wait_agent",
                        "senderThreadId": "thread-parent",
                        "receiverThreadIds": ["thread-a", "thread-b"],
                    }
                }
            },
        )

        tc = _parse_tool_call(update)

        assert tc.collaboration_tool == "wait_agent"
        assert tc.collaboration_receivers == ("thread-a", "thread-b")
        assert tc.collaboration_model == "gpt-test"
        assert tc.content == "等待卡片审计"
        assert tc.subagent_states == (
            {"source_id": "thread-a", "status": "running", "message": "正在核查普通卡"},
            {"source_id": "thread-b", "status": "completed", "message": "已核查 Deep 卡"},
        )

    def test_codex_collaboration_rejects_scalar_receiver_ids(self):
        update = MockToolCallStart(
            tool_call_id="collaboration-call-private",
            title="wait_agent",
            kind="other",
            status="in_progress",
            raw_input={
                "receiverThreadIds": "thread-a",
                "agentsStates": {
                    "thread-a": {"status": "running", "message": "正在核查"},
                },
            },
            field_meta={
                "codex": {
                    "collaboration": {
                        "tool": "wait_agent",
                        "receiverThreadIds": "thread-a",
                    }
                }
            },
        )

        tc = _parse_tool_call(update)

        assert tc.collaboration_receivers == ()
        assert tc.subagent_states == (
            {"source_id": "thread-a", "status": "running", "message": "正在核查"},
        )


class TestParsePlan:
    def test_basic(self):
        class MockAgentPlanUpdate:
            entries = [
                MockPlanEntry("Step 1", status="completed"),
                MockPlanEntry("Step 2", status="in_progress"),
            ]

        plan = _parse_plan(MockAgentPlanUpdate())
        assert len(plan.entries) == 2
        assert plan.entries[0].content == "Step 1"
        assert plan.entries[0].status == "completed"

    def test_skips_empty_entries(self):
        class MockAgentPlanUpdate:
            entries = [
                MockPlanEntry("", status="completed"),
                MockPlanEntry("   ", status="completed"),
                MockPlanEntry(None, status="completed"),
                MockPlanEntry("Real step", status="pending"),
            ]

        plan = _parse_plan(MockAgentPlanUpdate())
        assert [e.content for e in plan.entries] == ["Real step"]

    def test_empty_plan(self):
        class MockAgentPlanUpdate:
            entries = []

        plan = _parse_plan(MockAgentPlanUpdate())
        assert plan.entries == []


class TestGhostAPClient:
    def setup_method(self):
        self.events: list[ACPEvent] = []
        self.client = GhostAPClient(on_event=self.events.append)

    def _run_async(self, coro):
        """Run async coroutine in sync tests (Py3.12-safe)."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

    def test_request_permission_auto_approve(self):
        # Create mock options with an allow_once option
        mock_option = MagicMock()
        mock_option.kind = "allow_once"
        mock_option.option_id = "opt1"
        result = self._run_async(
            self.client.request_permission(
                options=[mock_option],
                session_id="s1",
                tool_call=MagicMock(),
            )
        )
        assert result.outcome.outcome == "selected"
        assert result.outcome.option_id == "opt1"

    def test_message_chunk_preserves_source_from_meta(self):
        """AgentMessageChunk source metadata is copied into ACPEvent."""
        from acp.schema import AgentMessageChunk, TextContentBlock

        from src.acp.models import ACPEventType

        update = AgentMessageChunk(
            sessionUpdate="agent_message_chunk",
            _meta={"source_id": "agent-a"},
            content=TextContentBlock(type="text", text="hello"),
        )

        self.client._handle_message_chunk(update)

        assert len(self.events) == 1
        assert self.events[0].event_type == ACPEventType.TEXT_CHUNK
        assert self.events[0].text == "hello"
        assert self.events[0].source_id == "agent-a"

    def test_message_chunk_ignores_generic_chunk_id_as_source(self):
        """Per-chunk ids must not split one assistant stream into text blocks."""
        from acp.schema import AgentMessageChunk, TextContentBlock

        update = AgentMessageChunk(
            sessionUpdate="agent_message_chunk",
            _meta={"id": "chunk-1"},
            content=TextContentBlock(type="text", text="现在让"),
        )

        self.client._handle_message_chunk(update)

        assert len(self.events) == 1
        assert self.events[0].source_id is None

    def test_tool_call_copies_codex_subagent_source_to_event(self):
        from acp.schema import ToolCallStart

        update = ToolCallStart.model_validate(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "activity-call-private",
                "title": "Start subagent card-audit",
                "kind": "other",
                "status": "in_progress",
                "rawInput": {
                    "agentThreadId": "thread-private",
                    "agentPath": "/root/card-audit",
                    "activityKind": "started",
                },
                "_meta": {
                    "codex": {
                        "subagent": {
                            "threadId": "thread-private",
                            "path": "/root/card-audit",
                            "activity": "started",
                        }
                    }
                },
            }
        )

        self.client._handle_tool_call_start(update)

        assert len(self.events) == 1
        assert self.events[0].source_id == "thread-private"
        assert self.events[0].tool_call is not None
        assert self.events[0].tool_call.subagent_activity == "started"


def test_read_write_text_file(tmp_path: Path):
    root = str(tmp_path)
    events: list[ACPEvent] = []
    client = GhostAPClient(on_event=events.append, root_dir=root)

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(client.write_text_file("s1", "a.txt", "hello"))
        resp = loop.run_until_complete(client.read_text_file("s1", "a.txt"))
        assert resp.content == "hello"
    finally:
        loop.close()


def test_tool_filter_blocks_acp_file_and_terminal_tools(tmp_path: Path):
    root = str(tmp_path)
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    client = GhostAPClient(on_event=lambda e: None, root_dir=root, sandbox=SandboxExecutor())
    client.set_tool_filter(lambda tool, args: False)

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        read_resp = loop.run_until_complete(client.read_text_file("s1", "a.txt"))
        write_resp = loop.run_until_complete(client.write_text_file("s1", "a.txt", "changed"))
        term_resp = loop.run_until_complete(client.create_terminal(command="echo hi", session_id="s1"))
        term_out = loop.run_until_complete(client.terminal_output(session_id="s1", terminal_id=term_resp.terminal_id))

        assert read_resp.content == ""
        assert read_resp.field_meta and read_resp.field_meta.get("blocked") is True
        assert write_resp and write_resp.field_meta and write_resp.field_meta.get("blocked") is True
        assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "hello"
        assert term_resp.field_meta and term_resp.field_meta.get("blocked") is True
        assert "工具权限" in term_out.output
    finally:
        loop.close()


def test_tool_filter_blocks_auto_approved_permission_request():
    client = GhostAPClient(on_event=lambda e: None, auto_approve=True)
    client.set_tool_filter(lambda tool, args: False)

    opt = MagicMock()
    opt.kind = "allow_once"
    opt.option_id = "opt1"
    tool_call = MagicMock()
    tool_call.kind = "execute"
    tool_call.raw_input = {"command": "echo hi"}

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        resp = loop.run_until_complete(client.request_permission(options=[opt], session_id="s1", tool_call=tool_call))
        assert resp.outcome.outcome == "cancelled"
    finally:
        loop.close()


@pytest.mark.parametrize(
    ("kind", "raw_input", "expected_tool", "expected_args"),
    [
        ("read", {"path": "/project/secret.txt"}, "file_read", {"path": "/project/secret.txt"}),
        ("edit", {"path": "/project/app.py"}, "file_write", {"path": "/project/app.py"}),
        ("delete", {"path": "/project/app.py"}, "file_write", {"path": "/project/app.py"}),
        (
            "move",
            {"source": "/project/a.py", "destination": "/project/b.py"},
            "file_write",
            {"source": "/project/a.py", "destination": "/project/b.py"},
        ),
        ("search", {"query": "password"}, "search", {"query": "password"}),
        ("fetch", {"url": "https://example.invalid"}, "fetch", {"url": "https://example.invalid"}),
        ("other", {"operation": "custom"}, "other", {"operation": "custom"}),
        ("think", {"topic": "plan"}, "think", {"topic": "plan"}),
        ("switch_mode", {"mode": "code"}, "switch_mode", {"mode": "code"}),
        ("execute", {"opaque": "missing command"}, "execute", {"opaque": "missing command"}),
        ("future_kind", {"operation": "future"}, "other", {"operation": "future"}),
        (None, {"operation": "unspecified"}, "other", {"operation": "unspecified"}),
    ],
)
def test_permission_bridge_checks_every_kind_before_auto_approve(
    kind,
    raw_input,
    expected_tool,
    expected_args,
):
    seen: list[tuple[str, dict]] = []
    client = GhostAPClient(on_event=lambda e: None, auto_approve=True)
    client.set_tool_filter(
        lambda tool, args: seen.append((tool, args or {})) or False
    )
    option = PermissionOption(
        optionId="allow-once",
        name="Allow once",
        kind="allow_once",
    )
    tool_call = SimpleNamespace(kind=kind, raw_input=raw_input)

    response = asyncio.run(
        client.request_permission(
            options=[option],
            session_id="s1",
            tool_call=tool_call,
        )
    )

    assert response.outcome.outcome == "cancelled"
    assert seen == [(expected_tool, expected_args)]


def test_permission_bridge_filter_exception_fails_closed_before_auto_approve():
    client = GhostAPClient(on_event=lambda e: None, auto_approve=True)

    def broken_filter(_tool, _args):
        raise RuntimeError("filter unavailable")

    client.set_tool_filter(broken_filter)
    option = PermissionOption(
        optionId="allow-once",
        name="Allow once",
        kind="allow_once",
    )

    response = asyncio.run(
        client.request_permission(
            options=[option],
            session_id="s1",
            tool_call=SimpleNamespace(
                kind="read",
                raw_input={"path": "/project/secret.txt"},
            ),
        )
    )

    assert response.outcome.outcome == "cancelled"


def test_permission_bridge_malformed_execute_command_remains_fail_closed():
    client = GhostAPClient(on_event=lambda e: None, auto_approve=True)
    option = PermissionOption(
        optionId="allow-once",
        name="Allow once",
        kind="allow_once",
    )

    response = asyncio.run(
        client.request_permission(
            options=[option],
            session_id="s1",
            tool_call=SimpleNamespace(
                kind="execute",
                raw_input={"command": ["rm", "-rf", "/"]},
            ),
        )
    )

    assert response.outcome.outcome == "cancelled"


def test_permission_bridge_classifies_canonical_git_execute_as_git():
    seen: list[tuple[str, dict]] = []
    client = GhostAPClient(on_event=lambda e: None, auto_approve=True)
    client.set_tool_filter(lambda tool, args: seen.append((tool, args or {})) or tool == "git")

    opt = MagicMock()
    opt.kind = "allow_once"
    opt.option_id = "opt1"
    tool_call = MagicMock()
    tool_call.kind = "execute"
    tool_call.raw_input = {"command": "/usr/bin/git status --short"}

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        resp = loop.run_until_complete(client.request_permission(options=[opt], session_id="s1", tool_call=tool_call))
        assert resp.outcome.outcome == "selected"
        assert seen == [
            ("git", {"command": "/usr/bin/git status --short", "cwd": client._root_dir})  # noqa: SLF001
        ]
    finally:
        loop.close()


def test_permission_bridge_keeps_wrapped_git_execute_on_shell_policy():
    seen: list[str] = []
    client = GhostAPClient(on_event=lambda e: None, auto_approve=True)
    client.set_tool_filter(lambda tool, args: seen.append(tool) or tool == "git")

    opt = MagicMock()
    opt.kind = "allow_once"
    opt.option_id = "opt1"
    tool_call = MagicMock()
    tool_call.kind = "execute"
    tool_call.raw_input = {"command": "sh -c 'git status'"}

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        resp = loop.run_until_complete(client.request_permission(options=[opt], session_id="s1", tool_call=tool_call))
        assert resp.outcome.outcome == "cancelled"
        assert seen == ["shell"]
    finally:
        loop.close()


def test_permission_bridge_fails_closed_when_safety_check_raises():
    sandbox = MagicMock(spec=SandboxExecutor)
    sandbox.is_command_safe.side_effect = RuntimeError("safety backend unavailable")
    client = GhostAPClient(on_event=lambda e: None, auto_approve=True, sandbox=sandbox)

    opt = MagicMock()
    opt.kind = "allow_once"
    opt.option_id = "opt1"
    tool_call = MagicMock()
    tool_call.kind = "execute"
    tool_call.raw_input = {"command": "echo hi"}

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        resp = loop.run_until_complete(client.request_permission(options=[opt], session_id="s1", tool_call=tool_call))
        assert resp.outcome.outcome == "cancelled"
    finally:
        loop.close()


class TestRebindThread:
    def _make_manager(self):
        from src.acp.manager import ACPSessionManager

        return ACPSessionManager("coco", session_timeout=999999)

    def test_rebind_moves_session(self):
        mgr = self._make_manager()
        sentinel = MagicMock()
        old_key = mgr._session_key("chat1", "proj1", thread_id=None)
        mgr._sessions[old_key] = sentinel

        result = mgr.rebind_thread("chat1", "proj1", "thread_resp_123")

        assert result is True
        new_key = mgr._session_key("chat1", "proj1", thread_id="thread_resp_123")
        assert mgr._sessions.get(new_key) is sentinel
        assert old_key not in mgr._sessions

    def test_rebind_returns_false_when_missing(self):
        mgr = self._make_manager()

        result = mgr.rebind_thread("chat1", "proj1", "thread_resp_456")

        assert result is False

    def test_rebind_does_not_affect_other_keys(self):
        mgr = self._make_manager()
        sentinel_a = MagicMock()
        sentinel_b = MagicMock()
        key_a = mgr._session_key("chat1", "projA", thread_id=None)
        key_b = mgr._session_key("chat1", "projB", thread_id=None)
        mgr._sessions[key_a] = sentinel_a
        mgr._sessions[key_b] = sentinel_b

        mgr.rebind_thread("chat1", "projA", "t1")

        assert mgr._sessions.get(mgr._session_key("chat1", "projA", thread_id="t1")) is sentinel_a
        assert mgr._sessions.get(key_b) is sentinel_b
        assert key_a not in mgr._sessions

    def test_rebind_cleans_existing_target_session(self):
        mgr = self._make_manager()
        old_session = MagicMock()
        old_session.session_id = "sid_old"
        old_session.message_count = 3
        existing_at_target = MagicMock()
        existing_at_target.session_id = "sid_target"
        existing_at_target.message_count = 5

        old_key = mgr._session_key("chat1", "proj1", thread_id=None)
        new_key = mgr._session_key("chat1", "proj1", thread_id="thread_new")
        mgr._sessions[old_key] = old_session
        mgr._sessions[new_key] = existing_at_target

        result = mgr.rebind_thread("chat1", "proj1", "thread_new")

        assert result is True
        assert mgr._sessions.get(new_key) is old_session
        assert old_key not in mgr._sessions
        existing_at_target.close.assert_called_once()
        existing_at_target.to_snapshot.assert_called_once()


def test_read_text_file_path_escape_denied(tmp_path: Path):
    root = str(tmp_path)
    client = GhostAPClient(on_event=lambda e: None, root_dir=root)
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        resp = loop.run_until_complete(client.read_text_file("s1", "../etc/passwd"))
        assert resp.content == ""
        assert resp.field_meta and "error" in resp.field_meta
    finally:
        loop.close()


def test_terminal_virtual_execution(tmp_path: Path):
    root = str(tmp_path)
    client = GhostAPClient(on_event=lambda e: None, root_dir=root, sandbox=SandboxExecutor())
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        create = loop.run_until_complete(client.create_terminal(command="echo hi", session_id="s1"))
        assert create.terminal_id.startswith("term_")

        out = loop.run_until_complete(client.terminal_output(session_id="s1", terminal_id=create.terminal_id))
        assert "hi" in out.output
        assert out.exit_status and out.exit_status.exit_code == 0
        assert out.truncated in (True, False)
    finally:
        loop.close()


def test_history_store_missing_file_returns_empty(tmp_path: Path):
    store = ACPHistoryStore(base_dir=str(tmp_path))
    assert store.load("no_such_session") == []


def test_history_store_skips_corrupt_lines(tmp_path: Path):
    store = ACPHistoryStore(base_dir=str(tmp_path))
    p = tmp_path / "s1.jsonl"
    p.write_text("{not json}\n" + '{"kind": "execute", "data": {"command": "echo hi"}}\n', encoding="utf-8")
    items = store.load("s1")
    assert len(items) == 1
    assert items[0]["kind"] == "execute"


def test_client_records_execute_history(tmp_path: Path):
    store = ACPHistoryStore(base_dir=str(tmp_path))
    client = GhostAPClient(
        on_event=lambda e: None, root_dir=str(tmp_path), sandbox=SandboxExecutor(), history_store=store
    )
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        resp = loop.run_until_complete(client.create_terminal(command="echo hi", session_id="s1"))
        assert resp.terminal_id
    finally:
        loop.close()

    items = store.load("s1")
    kinds = [x.get("kind") for x in items]
    assert "execute" in kinds


def test_permission_rejects_unsafe_execute():
    events: list[ACPEvent] = []
    client = GhostAPClient(on_event=events.append, auto_approve=True)
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)

        # allow_once option exists but should still be denied by safety policy
        opt = MagicMock()
        opt.kind = "allow_once"
        opt.option_id = "opt1"

        tool_call = MagicMock()
        tool_call.kind = "execute"
        tool_call.raw_input = {"command": "rm -rf /"}

        resp = loop.run_until_complete(client.request_permission(options=[opt], session_id="s1", tool_call=tool_call))
        assert resp.outcome.outcome == "cancelled"
    finally:
        loop.close()


def test_resolve_agent_spec_coco_has_command():
    if not shutil.which("coco"):
        pytest.skip("coco binary not available")
    cmd, args = resolve_agent_spec("coco")
    assert cmd == "coco"
    assert args == ["acp", "serve"]


def test_read_text_file_truncates(tmp_path: Path):
    root = str(tmp_path)
    client = GhostAPClient(on_event=lambda e: None, root_dir=root)
    big = "x" * 300_000
    (tmp_path / "big.txt").write_text(big, encoding="utf-8")

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        resp = loop.run_until_complete(client.read_text_file("s1", "big.txt"))
        assert len(resp.content) == 200_000
        assert resp.field_meta and resp.field_meta.get("truncated") is True
    finally:
        loop.close()


def test_acp_011_permission_arguments_keep_allow_once_selection():
    client = GhostAPClient(on_event=lambda _event: None, auto_approve=True)
    option = MagicMock()
    option.kind = "allow_once"
    option.option_id = "allow-once"

    response = asyncio.run(
        client.request_permission("session-1", MagicMock(), [option])
    )

    assert response.outcome.outcome == "selected"
    assert response.outcome.option_id == "allow-once"


def test_acp_011_read_and_write_argument_order_and_line_limit(tmp_path: Path):
    client = GhostAPClient(on_event=lambda _event: None, root_dir=str(tmp_path))

    asyncio.run(client.write_text_file("session-1", "lines.txt", "one\ntwo\nthree\n"))
    response = asyncio.run(
        client.read_text_file("session-1", "lines.txt", line=2, limit=1)
    )

    assert response.content == "two\n"


def test_acp_011_terminal_arguments_honor_env_cwd_and_output_byte_limit(tmp_path: Path):
    client = GhostAPClient(on_event=lambda _event: None, root_dir=str(tmp_path))

    terminal = asyncio.run(
        client.create_terminal(
            "session-1",
            "sh",
            ["-c", "printf '%s' \"$ACP_TEST_VALUE-abcd\""],
            [EnvVariable(name="ACP_TEST_VALUE", value="value")],
            str(tmp_path),
            2,
        )
    )
    output = asyncio.run(
        client.terminal_output("session-1", terminal.terminal_id)
    )

    assert output.output == "cd"
    assert output.truncated is True


def test_acp_011_terminal_rejects_cwd_outside_project_root(tmp_path: Path):
    client = GhostAPClient(on_event=lambda _event: None, root_dir=str(tmp_path))

    terminal = asyncio.run(
        client.create_terminal("session-1", "pwd", None, None, "/tmp", None)
    )
    output = asyncio.run(
        client.terminal_output("session-1", terminal.terminal_id)
    )

    assert terminal.field_meta and terminal.field_meta.get("blocked") is True
    assert "工作目录" in output.output
