from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

SCRIPT = Path("scripts/validate_workflow_tenant.py")
CHECK_NAMES = {
    "origin_recipient_bound",
    "historical_page_hash_unchanged_after_freeze",
    "terminal_all_results_visible",
    "terminal_has_no_stop_action",
    "desktop_pages_contiguous",
    "mobile_pages_contiguous",
}
ARTIFACT_NAMES = {
    "redacted_event_log_sha256",
    "desktop_screenshot_sha256",
    "mobile_screenshot_sha256",
}


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _binding_args() -> list[str]:
    return [
        "--run-id",
        "wf-live-run-001",
        "--service-instance-id",
        "ghostap-staging-01",
        "--tenant-hash",
        _digest("tenant"),
        "--chat-id-hash",
        _digest("chat"),
        "--expected-result-count",
        "3",
    ]


def _run(*args: str, live: bool = False) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("GHOSTAP_WORKFLOW_ACCEPTANCE_LIVE", None)
    if live:
        env["GHOSTAP_WORKFLOW_ACCEPTANCE_LIVE"] = "1"
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args, *_binding_args()],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _create_template(path: Path) -> dict[str, object]:
    result = _run("--template-out", str(path))
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(path.read_text(encoding="utf-8"))


def _passing_capture(path: Path) -> None:
    capture = _create_template(path)
    page_hashes = [_digest(f"message-{page}") for page in range(3)]
    events = [
        ("create", 0),
        ("patch", 0),
        ("create", 1),
        ("freeze", 0),
        ("patch", 1),
        ("create", 2),
        ("freeze", 1),
        ("patch", 2),
        ("finish", 2),
    ]
    capture["events"] = [
        {
            "sequence": sequence,
            "kind": kind,
            "page": page,
            "message_id_hash": page_hashes[page],
            "payload_sha256": _digest(f"payload-{sequence}"),
            "captured_at": 1_786_224_000 + sequence,
        }
        for sequence, (kind, page) in enumerate(events, start=1)
    ]
    capture["checks"] = {name: True for name in CHECK_NAMES}
    capture["artifacts"] = {name: _digest(name) for name in ARTIFACT_NAMES}
    capture["observed_result_count"] = 3
    capture["attestor"] = "release-operator"
    path.write_text(json.dumps(capture), encoding="utf-8")


def test_template_is_exclusive_private_and_fail_closed(tmp_path: Path) -> None:
    capture_path = tmp_path / "workflow-live-capture.json"

    capture = _create_template(capture_path)

    assert stat.S_IMODE(capture_path.stat().st_mode) == 0o600
    assert capture["schema_version"] == 1
    assert capture["events"] == []
    assert capture["checks"] == {name: False for name in CHECK_NAMES}
    assert capture["artifacts"] == {name: "" for name in ARTIFACT_NAMES}
    assert capture["observed_result_count"] == 0
    assert capture["attestor"] == ""
    duplicate = _run("--template-out", str(capture_path))
    assert duplicate.returncode == 1
    assert json.loads(duplicate.stdout)["status"] == "failed"


def test_live_evaluation_requires_explicit_opt_in(tmp_path: Path) -> None:
    capture_path = tmp_path / "workflow-live-capture.json"
    _passing_capture(capture_path)

    result = _run("--live", "--live-results", str(capture_path))

    assert result.returncode == 1
    output = json.loads(result.stdout)
    assert output["status"] == "failed"
    assert output["reason"] == "live mode requires GHOSTAP_WORKFLOW_ACCEPTANCE_LIVE=1"


def test_complete_bound_multi_page_capture_passes(tmp_path: Path) -> None:
    capture_path = tmp_path / "workflow-live-capture.json"
    _passing_capture(capture_path)

    result = _run("--live", "--live-results", str(capture_path), live=True)

    assert result.returncode == 0, result.stdout + result.stderr
    output = json.loads(result.stdout)
    assert output == {
        "event_count": 9,
        "live_mode": True,
        "page_count": 3,
        "run_id": "wf-live-run-001",
        "status": "passed",
    }


def test_freezing_a_page_before_its_successor_is_visible_fails(tmp_path: Path) -> None:
    capture_path = tmp_path / "workflow-live-capture.json"
    _passing_capture(capture_path)
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    capture["events"][2], capture["events"][3] = capture["events"][3], capture["events"][2]
    for sequence, event in enumerate(capture["events"], start=1):
        event["sequence"] = sequence
    capture_path.write_text(json.dumps(capture), encoding="utf-8")

    result = _run("--live", "--live-results", str(capture_path), live=True)

    assert result.returncode == 1
    output = json.loads(result.stdout)
    assert output["status"] == "failed"
    assert "visible before page 0 is frozen" in output["reason"]


def test_capture_cannot_be_rebound_to_another_run(tmp_path: Path) -> None:
    capture_path = tmp_path / "workflow-live-capture.json"
    _passing_capture(capture_path)
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    capture["binding"]["run_id"] = "wf-live-run-other"
    capture_path.write_text(json.dumps(capture), encoding="utf-8")

    result = _run("--live", "--live-results", str(capture_path), live=True)

    assert result.returncode == 1
    output = json.loads(result.stdout)
    assert output["status"] == "failed"
    assert output["reason"] == "live capture binding does not match requested run"
