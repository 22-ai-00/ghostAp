import json
import os
import stat
import subprocess
import sys
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import src.autonomous.journal.writer as writer_module
from src.autonomous.journal import (
    AnchorMismatchError,
    CommitState,
    JournalClosedError,
    JournalWriter,
    MemoryAnchor,
    WriterLockError,
)
from src.autonomous.journal.frame import (
    GENESIS_HASH,
    JournalEvent,
    JournalIntegrityError,
    TransactionFrame,
)

HMAC_KEY = b"journal-writer-test-key-at-least-32-bytes"
JOURNAL_NAME = "journal.jsonl"


def event(
    aggregate_id: str = "goal_1",
    *,
    event_type: str = "goal.created",
    value: str = "durable",
) -> JournalEvent:
    return JournalEvent(
        event_type=event_type,
        aggregate_id=aggregate_id,
        payload={"value": value},
    )


def open_writer(
    base_dir: Path,
    anchor: Any,
    *,
    writer_epoch: int = 7,
    **kwargs: Any,
) -> JournalWriter:
    return JournalWriter.open(
        base_dir,
        anchor=anchor,
        hmac_key=HMAC_KEY,
        writer_epoch=writer_epoch,
        **kwargs,
    )


def journal_path(base_dir: Path) -> Path:
    return base_dir / JOURNAL_NAME


def committed_frame(result: Any) -> TransactionFrame:
    return getattr(result, "frame", result)


def commit_state(result: Any) -> CommitState:
    frame = committed_frame(result)
    state = getattr(result, "state", None)
    if state is None:
        state = getattr(result, "commit_state", None)
    if state is None:
        state = getattr(frame, "anchor_state", None)
    return state


def anchor_position(anchor: Any) -> tuple[int, str]:
    position = anchor.read()
    if isinstance(position, tuple):
        return position
    if isinstance(position, Mapping):
        return int(position["sequence"]), str(position["frame_hash"])
    return int(position.sequence), str(position.frame_hash)


def records(path: Path) -> list[bytes]:
    return path.read_bytes().splitlines()


def rewrite_record(path: Path, index: int, transform: Any) -> None:
    current = path.read_bytes().splitlines(keepends=True)
    newline = b"\n" if current[index].endswith(b"\n") else b""
    record = current[index].removesuffix(newline)
    current[index] = transform(record) + newline
    path.write_bytes(b"".join(current))


def mutate_json_field(record: bytes, field: str, value: Any) -> bytes:
    decoded = json.loads(record)
    decoded[field] = value
    return json.dumps(decoded, sort_keys=True, separators=(",", ":")).encode()


def corrupt_hmac(record: bytes) -> bytes:
    decoded = json.loads(record)
    digest = decoded["hmac_digest"]
    replacement = "0" if digest[0] != "0" else "1"
    decoded["hmac_digest"] = replacement + digest[1:]
    encoded = json.dumps(decoded, sort_keys=True, separators=(",", ":")).encode()
    assert len(encoded) == len(record)
    return encoded


class RejectingAnchor:
    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def read(self) -> Any:
        return self.delegate.read()

    def compare_and_swap(self, *args: Any, **kwargs: Any) -> bool:
        self.calls.append((args, kwargs))
        return False


class RecordingAnchor:
    def __init__(self, delegate: Any, timeline: list[str]) -> None:
        self.delegate = delegate
        self.timeline = timeline

    def read(self) -> Any:
        return self.delegate.read()

    def compare_and_swap(self, *args: Any, **kwargs: Any) -> bool:
        self.timeline.append("anchor_cas")
        return self.delegate.compare_and_swap(*args, **kwargs)


class RaisingAnchor:
    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate

    def read(self) -> Any:
        return self.delegate.read()

    def compare_and_swap(self, *_args: Any, **_kwargs: Any) -> bool:
        raise OSError("anchor unavailable")


class RecordingFsOps:
    def __init__(self, timeline: list[str]) -> None:
        self.timeline = timeline

    def fsync_file(self, file_or_fd: Any) -> None:
        self.timeline.append("file_fsync")
        fd = file_or_fd if isinstance(file_or_fd, int) else file_or_fd.fileno()
        os.fsync(fd)

    def fsync_directory(self, directory: str | Path) -> None:
        self.timeline.append("dir_fsync")
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class FailingFsOps:
    def __init__(self, boundary: str) -> None:
        self.boundary = boundary

    def fsync_file(self, file_or_fd: Any) -> None:
        if self.boundary == "file_fsync":
            raise OSError("injected file fsync failure")
        fd = file_or_fd if isinstance(file_or_fd, int) else file_or_fd.fileno()
        os.fsync(fd)

    def fsync_directory(self, directory: str | Path) -> None:
        if self.boundary == "dir_fsync":
            raise OSError("injected directory fsync failure")
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class BlockingFsOps:
    def __init__(self) -> None:
        self.write_started = threading.Event()
        self.allow_fsync = threading.Event()

    def fsync_file(self, file_or_fd: Any) -> None:
        self.write_started.set()
        assert self.allow_fsync.wait(timeout=5)
        fd = file_or_fd if isinstance(file_or_fd, int) else file_or_fd.fileno()
        os.fsync(fd)

    def fsync_directory(self, directory: str | Path) -> None:
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def test_second_writer_fails_nonblocking_and_close_releases_lock(
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "journal"
    anchor = MemoryAnchor()
    first = open_writer(base_dir, anchor)

    try:
        with pytest.raises(WriterLockError):
            open_writer(base_dir, anchor, writer_epoch=8)
    finally:
        first.close()

    with open_writer(base_dir, anchor, writer_epoch=8) as reopened:
        assert list(reopened.replay(from_sequence=1)) == []


def test_second_process_cannot_acquire_writer_lock(tmp_path: Path) -> None:
    base_dir = tmp_path / "journal"
    writer = open_writer(base_dir, MemoryAnchor())
    code = f"""
from pathlib import Path
from src.autonomous.journal import JournalWriter, MemoryAnchor, WriterLockError
try:
    JournalWriter.open(
        Path({str(base_dir)!r}),
        anchor=MemoryAnchor(),
        hmac_key={HMAC_KEY!r},
        writer_epoch=8,
    )
except WriterLockError:
    raise SystemExit(0)
raise SystemExit(1)
"""

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    finally:
        writer.close()

    assert result.returncode == 0, result.stderr


def test_frames_start_at_genesis_and_chain_sequence_hash_and_writer_epoch(
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "journal"
    anchor = MemoryAnchor()
    writer = open_writer(base_dir, anchor, writer_epoch=41)

    first = committed_frame(writer.commit([event()], {"goal_1": 0}))
    second = committed_frame(
        writer.commit(
            [event(event_type="goal.updated", value="second")],
            {"goal_1": 1},
        )
    )
    writer.close()

    assert first.sequence == 1
    assert first.previous_hash == GENESIS_HASH
    assert first.writer_epoch == 41
    assert second.sequence == 2
    assert second.previous_hash == first.frame_hash
    assert second.writer_epoch == 41

    with open_writer(base_dir, anchor, writer_epoch=42) as restarted:
        third = committed_frame(
            restarted.commit(
                [event(event_type="goal.updated", value="third")],
                {"goal_1": 2},
            )
        )

    assert third.sequence == 3
    assert third.previous_hash == second.frame_hash
    assert third.writer_epoch == 42


def test_expected_aggregate_version_mismatch_does_not_append(
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "journal"
    writer = open_writer(base_dir, MemoryAnchor())
    writer.commit([event()], {"goal_1": 0})
    before = journal_path(base_dir).read_bytes()

    with pytest.raises(JournalIntegrityError, match="version"):
        writer.commit(
            [event(event_type="goal.updated", value="stale")],
            {"goal_1": 0},
        )

    assert journal_path(base_dir).read_bytes() == before
    assert [frame.sequence for frame in writer.replay(from_sequence=1)] == [1]
    writer.close()


def test_successful_anchor_cas_advances_high_water_mark(
    tmp_path: Path,
) -> None:
    anchor = MemoryAnchor()
    with open_writer(tmp_path / "journal", anchor) as writer:
        result = writer.commit([event()], {"goal_1": 0})
        frame = committed_frame(result)

    assert commit_state(result) is CommitState.ANCHORED
    assert anchor_position(anchor) == (frame.sequence, frame.frame_hash)


def test_anchor_cas_failure_is_durable_but_closes_writes_and_restart_fails_closed(
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "journal"
    durable_anchor = MemoryAnchor()
    rejecting_anchor = RejectingAnchor(durable_anchor)
    writer = open_writer(base_dir, rejecting_anchor)

    result = writer.commit([event()], {"goal_1": 0})
    frame = committed_frame(result)

    assert frame.sequence == 1
    assert commit_state(result) is CommitState.DURABLE_NOT_ANCHORED
    assert len(rejecting_anchor.calls) == 1
    assert anchor_position(durable_anchor) == (0, GENESIS_HASH)
    assert [item.sequence for item in writer.replay(from_sequence=1)] == [1]

    with pytest.raises(JournalClosedError):
        writer.commit(
            [event(event_type="goal.updated", value="must-not-write")],
            {"goal_1": 1},
        )

    writer.close()
    with pytest.raises(AnchorMismatchError):
        open_writer(base_dir, durable_anchor, writer_epoch=8)


def test_anchor_exception_is_durable_but_closes_writes(
    tmp_path: Path,
) -> None:
    durable_anchor = MemoryAnchor()
    writer = open_writer(tmp_path / "journal", RaisingAnchor(durable_anchor))

    result = writer.commit([event()], {"goal_1": 0})

    assert commit_state(result) is CommitState.DURABLE_NOT_ANCHORED
    with pytest.raises(JournalClosedError):
        writer.commit(
            [event(event_type="goal.updated", value="must-not-write")],
            {"goal_1": 1},
        )
    writer.close()


def test_short_append_fails_closed_before_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = open_writer(tmp_path / "journal", MemoryAnchor())
    original_append = writer_module._append_record

    def short_append(path: Path, record: bytes, fs_ops: Any) -> None:
        with open(path, "ab", buffering=0) as file:
            file.write(record[: len(record) // 2])
            fs_ops.fsync_file(file)
        raise OSError("short journal write")

    monkeypatch.setattr(writer_module, "_append_record", short_append)

    with pytest.raises(OSError, match="short journal write"):
        writer.commit([event()], {"goal_1": 0})
    with pytest.raises(JournalClosedError):
        writer.commit([event()], {"goal_1": 0})

    monkeypatch.setattr(writer_module, "_append_record", original_append)
    writer.close()


def test_wrong_hmac_key_cannot_open_existing_journal(tmp_path: Path) -> None:
    base_dir = tmp_path / "journal"
    anchor = MemoryAnchor()
    writer = open_writer(base_dir, anchor)
    writer.commit([event()], {"goal_1": 0})
    writer.close()

    with pytest.raises(JournalIntegrityError, match="hmac"):
        JournalWriter.open(
            base_dir,
            anchor=anchor,
            hmac_key=b"different-journal-key-at-least-32-bytes",
            writer_epoch=8,
        )


def test_writer_enforces_private_directory_and_journal_permissions(
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "journal"
    base_dir.mkdir(mode=0o777)
    base_dir.chmod(0o777)

    with open_writer(base_dir, MemoryAnchor()):
        pass

    assert stat.S_IMODE(base_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(journal_path(base_dir).stat().st_mode) == 0o600


def test_commit_fsyncs_file_then_directory_before_anchor_cas(
    tmp_path: Path,
) -> None:
    timeline: list[str] = []
    anchor = RecordingAnchor(MemoryAnchor(), timeline)
    fs_ops = RecordingFsOps(timeline)
    writer = open_writer(tmp_path / "journal", anchor, fs_ops=fs_ops)
    timeline.clear()

    writer.commit([event()], {"goal_1": 0})

    assert timeline == ["file_fsync", "dir_fsync", "anchor_cas"]
    writer.close()


@pytest.mark.parametrize("boundary", ["file_fsync", "dir_fsync"])
def test_durability_boundary_failure_closes_writer(
    tmp_path: Path,
    boundary: str,
) -> None:
    writer = open_writer(
        tmp_path / "journal",
        MemoryAnchor(),
        fs_ops=FailingFsOps(boundary),
    )

    with pytest.raises(OSError, match="injected"):
        writer.commit([event()], {"goal_1": 0})

    with pytest.raises(JournalClosedError):
        writer.commit([event()], {"goal_1": 0})
    writer.close()


def test_replay_waits_for_inflight_commit_instead_of_recovering_its_tail(
    tmp_path: Path,
) -> None:
    fs_ops = BlockingFsOps()
    writer = open_writer(tmp_path / "journal", MemoryAnchor(), fs_ops=fs_ops)
    commit_done = threading.Event()
    replay_done = threading.Event()
    replayed: list[int] = []

    def commit() -> None:
        writer.commit([event()], {"goal_1": 0})
        commit_done.set()

    def replay() -> None:
        replayed.extend(frame.sequence for frame in writer.replay())
        replay_done.set()

    commit_thread = threading.Thread(target=commit)
    replay_thread = threading.Thread(target=replay)
    commit_thread.start()
    assert fs_ops.write_started.wait(timeout=5)
    replay_thread.start()

    assert not replay_done.wait(timeout=0.1)
    fs_ops.allow_fsync.set()
    commit_thread.join(timeout=5)
    replay_thread.join(timeout=5)

    assert commit_done.is_set()
    assert replay_done.is_set()
    assert replayed == [1]
    writer.close()


def test_close_waits_for_inflight_commit_before_releasing_writer_lock(
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "journal"
    fs_ops = BlockingFsOps()
    writer = open_writer(base_dir, MemoryAnchor(), fs_ops=fs_ops)
    commit_thread = threading.Thread(
        target=lambda: writer.commit([event()], {"goal_1": 0})
    )
    close_thread = threading.Thread(target=writer.close)
    commit_thread.start()
    assert fs_ops.write_started.wait(timeout=5)
    close_thread.start()

    assert close_thread.is_alive()
    with pytest.raises(WriterLockError):
        open_writer(base_dir, MemoryAnchor(), writer_epoch=8)

    fs_ops.allow_fsync.set()
    commit_thread.join(timeout=5)
    close_thread.join(timeout=5)
    assert not commit_thread.is_alive()
    assert not close_thread.is_alive()


def test_physically_truncated_tail_is_removed_without_losing_committed_frames(
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "journal"
    anchor = MemoryAnchor()
    writer = open_writer(base_dir, anchor)
    writer.commit([event()], {"goal_1": 0})
    writer.commit(
        [event(event_type="goal.updated", value="second")],
        {"goal_1": 1},
    )
    writer.close()
    path = journal_path(base_dir)
    committed_bytes = path.read_bytes()
    path.write_bytes(committed_bytes + b'{"magic":"GHOSTAP-JOURNAL","sequence":3')

    with open_writer(base_dir, anchor, writer_epoch=8) as recovered:
        assert [frame.sequence for frame in recovered.replay(from_sequence=1)] == [
            1,
            2,
        ]

    assert path.read_bytes() == committed_bytes


def test_anchor_confirmed_incomplete_tail_is_not_truncated(
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "journal"
    anchor = MemoryAnchor()
    writer = open_writer(base_dir, anchor)
    first = committed_frame(writer.commit([event()], {"goal_1": 0}))
    writer.close()
    path = journal_path(base_dir)
    original = path.read_bytes()
    path.write_bytes(original[:-12])

    with pytest.raises(AnchorMismatchError):
        open_writer(base_dir, anchor, writer_epoch=8)

    assert path.read_bytes() == original[:-12]
    assert anchor_position(anchor) == (first.sequence, first.frame_hash)


def test_explicit_uncommitted_tail_is_removed(
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "journal"
    anchor = MemoryAnchor()
    writer = open_writer(base_dir, anchor)
    first = committed_frame(writer.commit([event()], {"goal_1": 0}))
    writer.close()
    path = journal_path(base_dir)
    committed_bytes = path.read_bytes()
    uncommitted = TransactionFrame.seal(
        tx_id="tx_uncommitted",
        sequence=2,
        writer_epoch=8,
        timestamp=1_750_000_001.0,
        expected_versions={"goal_1": 1},
        aggregate_versions={"goal_1": 2},
        previous_hash=first.frame_hash,
        events=(event(event_type="goal.updated", value="uncommitted"),),
        hmac_key=HMAC_KEY,
        committed=False,
    )
    path.write_bytes(committed_bytes + uncommitted.to_bytes() + b"\n")

    with open_writer(base_dir, anchor, writer_epoch=8) as recovered:
        assert [frame.sequence for frame in recovered.replay(from_sequence=1)] == [1]

    assert path.read_bytes() == committed_bytes


@pytest.mark.parametrize(
    "transform",
    [
        pytest.param(lambda _: b'{"broken":', id="json"),
        pytest.param(corrupt_hmac, id="hmac"),
        pytest.param(
            lambda record: mutate_json_field(record, "sequence", 9),
            id="sequence",
        ),
    ],
)
def test_middle_corruption_raises_without_truncating_later_frames(
    tmp_path: Path,
    transform: Any,
) -> None:
    base_dir = tmp_path / "journal"
    anchor = MemoryAnchor()
    writer = open_writer(base_dir, anchor)
    writer.commit([event()], {"goal_1": 0})
    writer.commit(
        [event(event_type="goal.updated", value="second")],
        {"goal_1": 1},
    )
    writer.commit(
        [event(event_type="goal.updated", value="third")],
        {"goal_1": 2},
    )
    writer.close()
    path = journal_path(base_dir)
    rewrite_record(path, 1, transform)
    corrupted_bytes = path.read_bytes()
    assert len(records(path)) == 3

    with pytest.raises(JournalIntegrityError):
        open_writer(base_dir, anchor, writer_epoch=8)

    assert path.read_bytes() == corrupted_bytes
    assert len(records(path)) == 3


def test_replay_validates_skipped_prefix_and_complete_hash_chain(
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "journal"
    writer = open_writer(base_dir, MemoryAnchor())
    writer.commit([event()], {"goal_1": 0})
    writer.commit(
        [event(event_type="goal.updated", value="second")],
        {"goal_1": 1},
    )
    rewrite_record(journal_path(base_dir), 0, corrupt_hmac)

    try:
        with pytest.raises(JournalIntegrityError):
            list(writer.replay(from_sequence=2))
    finally:
        writer.close()


def test_replay_rejects_event_and_version_key_mismatch(tmp_path: Path) -> None:
    base_dir = tmp_path / "journal"
    base_dir.mkdir(mode=0o700)
    frame = TransactionFrame.seal(
        tx_id="tx_mismatch",
        sequence=1,
        writer_epoch=7,
        timestamp=1_750_000_000.0,
        expected_versions={"other": 0},
        aggregate_versions={"other": 1},
        previous_hash=GENESIS_HASH,
        events=(event("goal_1"),),
        hmac_key=HMAC_KEY,
    )
    journal_path(base_dir).write_bytes(frame.to_bytes())
    anchor = MemoryAnchor()
    assert anchor.compare_and_swap(0, GENESIS_HASH, 1, frame.frame_hash)

    with pytest.raises(JournalIntegrityError, match="aggregate"):
        open_writer(base_dir, anchor)
