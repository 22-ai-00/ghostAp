"""Regression tests for the production file-backed Journal anchor."""

from __future__ import annotations

import json
import os
import stat
import threading
from pathlib import Path

import pytest

from src.autonomous.journal.anchor import (
    AnchorCorruptionError,
    AnchorState,
    FileAnchor,
)
from src.autonomous.journal.frame import GENESIS_HASH


def test_file_anchor_compare_and_swap_is_shared_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "anchor.json"
    first = FileAnchor(path)
    second = FileAnchor(path)
    next_hash = "1" * 64

    assert first.read() == AnchorState()
    assert first.compare_and_swap(0, GENESIS_HASH, 1, next_hash) is True
    assert second.read() == AnchorState(1, next_hash)
    assert second.compare_and_swap(0, GENESIS_HASH, 1, "2" * 64) is False


def test_file_anchor_serializes_competing_cross_instance_cas(tmp_path: Path) -> None:
    path = tmp_path / "anchor.json"
    anchors = (FileAnchor(path), FileAnchor(path))
    barrier = threading.Barrier(3)
    outcomes: list[bool] = []

    def compete(anchor: FileAnchor, frame_hash: str) -> None:
        barrier.wait()
        outcomes.append(
            anchor.compare_and_swap(0, GENESIS_HASH, 1, frame_hash)
        )

    threads = [
        threading.Thread(target=compete, args=(anchors[0], "a" * 64)),
        threading.Thread(target=compete, args=(anchors[1], "b" * 64)),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == [False, True]
    assert anchors[0].read().sequence == 1


def test_file_anchor_atomically_replaces_and_fsyncs_file_then_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "anchor.json"
    timeline: list[str] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def record_fsync(fd: int) -> None:
        mode = os.fstat(fd).st_mode
        timeline.append("dir_fsync" if stat.S_ISDIR(mode) else "file_fsync")
        real_fsync(fd)

    def record_replace(src: object, dst: object, *args: object, **kwargs: object) -> None:
        timeline.append("replace")
        real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "fsync", record_fsync)
    monkeypatch.setattr(os, "replace", record_replace)

    anchor = FileAnchor(path)
    assert anchor.compare_and_swap(0, GENESIS_HASH, 1, "c" * 64) is True

    assert timeline[-3:] == ["file_fsync", "replace", "dir_fsync"]
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_file_anchor_confirms_replaced_state_after_directory_fsync_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "anchor.json"
    anchor = FileAnchor(path)
    real_fsync = os.fsync
    fail_directory_once = True

    def transient_fsync(fd: int) -> None:
        nonlocal fail_directory_once
        if fail_directory_once and stat.S_ISDIR(os.fstat(fd).st_mode):
            fail_directory_once = False
            raise OSError("transient directory fsync failure")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", transient_fsync)
    state = AnchorState(1, "c" * 64)
    with pytest.raises(OSError, match="directory fsync"):
        anchor.compare_and_swap(0, GENESIS_HASH, state.sequence, state.frame_hash)
    assert anchor.confirm_durable(state) is True
    assert anchor.confirm_durable(AnchorState(1, "d" * 64)) is False


@pytest.mark.parametrize(
    "payload",
    [
        b"not-json\n",
        b'{"version":1,"version":1,"sequence":0,"frame_hash":"' + b"0" * 64 + b'"}',
        json.dumps({"version": 1, "sequence": -1, "frame_hash": "0" * 64}).encode(),
        json.dumps({"version": 1, "sequence": 0, "frame_hash": "bad"}).encode(),
        json.dumps(
            {
                "version": 1,
                "sequence": 0,
                "frame_hash": GENESIS_HASH,
                "extra": True,
            }
        ).encode(),
    ],
)
def test_file_anchor_corruption_fails_closed(
    tmp_path: Path,
    payload: bytes,
) -> None:
    path = tmp_path / "anchor.json"
    path.write_bytes(payload)
    anchor = FileAnchor(path)

    with pytest.raises(AnchorCorruptionError):
        anchor.read()
    with pytest.raises(AnchorCorruptionError):
        anchor.compare_and_swap(0, GENESIS_HASH, 1, "d" * 64)
