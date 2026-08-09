#!/usr/bin/env python3
"""Create and evaluate redacted Workflow multi-card real-tenant evidence."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_EVENT_FIELDS = {
    "sequence",
    "kind",
    "page",
    "message_id_hash",
    "payload_sha256",
    "captured_at",
}
_EVENT_KINDS = {"create", "patch", "freeze", "finish"}
_CHECK_NAMES = {
    "origin_recipient_bound",
    "historical_page_hash_unchanged_after_freeze",
    "terminal_all_results_visible",
    "terminal_has_no_stop_action",
    "desktop_pages_contiguous",
    "mobile_pages_contiguous",
}
_ARTIFACT_NAMES = {
    "redacted_event_log_sha256",
    "desktop_screenshot_sha256",
    "mobile_screenshot_sha256",
}
_ENV = {
    "run_id": "GHOSTAP_WORKFLOW_RUN_ID",
    "service_instance_id": "GHOSTAP_WORKFLOW_SERVICE_INSTANCE_ID",
    "tenant_hash": "GHOSTAP_WORKFLOW_TENANT_HASH",
    "chat_id_hash": "GHOSTAP_WORKFLOW_CHAT_ID_HASH",
    "expected_result_count": "GHOSTAP_WORKFLOW_EXPECTED_RESULT_COUNT",
}


@dataclass(frozen=True)
class WorkflowTenantBinding:
    run_id: str
    service_instance_id: str
    tenant_hash: str
    chat_id_hash: str
    expected_result_count: int

    def __post_init__(self) -> None:
        for field_name in ("run_id", "service_instance_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
                raise ValueError(f"invalid {field_name.replace('_', '-')}")
        for field_name in ("tenant_hash", "chat_id_hash"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
                raise ValueError(f"invalid {field_name.replace('_', '-')}")
        if (
            isinstance(self.expected_result_count, bool)
            or not isinstance(self.expected_result_count, int)
            or self.expected_result_count < 1
        ):
            raise ValueError("expected-result-count must be a positive integer")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed Workflow multi-card tenant evidence evaluator. The default mode "
            "does not contact Lark; --live only evaluates an already redacted capture."
        )
    )
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--live-results", type=Path)
    parser.add_argument(
        "--template-out",
        type=Path,
        help="exclusively create a tenant-bound capture checklist with all checks false",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--service-instance-id")
    parser.add_argument("--tenant-hash")
    parser.add_argument("--chat-id-hash")
    parser.add_argument("--expected-result-count", type=int)
    return parser


def _emit(*, status: str, live_mode: bool, **values: Any) -> None:
    print(
        json.dumps(
            {"status": status, "live_mode": live_mode, **values},
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _binding(args: argparse.Namespace) -> tuple[WorkflowTenantBinding | None, list[str]]:
    values: dict[str, str | int] = {}
    missing: list[str] = []
    for field_name, env_name in _ENV.items():
        value = getattr(args, field_name)
        if value is None:
            value = os.environ.get(env_name, "")
        if value == "":
            missing.append(env_name)
        values[field_name] = value
    if missing:
        return None, missing
    try:
        expected_result_count = int(values["expected_result_count"])
    except (TypeError, ValueError) as exc:
        raise ValueError("expected-result-count must be a positive integer") from exc
    return (
        WorkflowTenantBinding(
            run_id=str(values["run_id"]),
            service_instance_id=str(values["service_instance_id"]),
            tenant_hash=str(values["tenant_hash"]),
            chat_id_hash=str(values["chat_id_hash"]),
            expected_result_count=expected_result_count,
        ),
        [],
    )


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        offset += os.write(fd, payload[offset:])


def _write_template(path: Path, binding: WorkflowTenantBinding) -> None:
    envelope = {
        "schema_version": 1,
        "binding": asdict(binding),
        "events": [],
        "checks": {name: False for name in sorted(_CHECK_NAMES)},
        "artifacts": {name: "" for name in sorted(_ARTIFACT_NAMES)},
        "observed_result_count": 0,
        "attestor": "",
    }
    payload = (json.dumps(envelope, ensure_ascii=False, indent=2) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        _write_all(fd, payload)
        os.fsync(fd)
        os.fchmod(fd, 0o600)
    finally:
        os.close(fd)


def _require_hash(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _validate_events(raw_events: object) -> tuple[int, int]:
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError("live capture must contain a non-empty event ledger")

    created: set[int] = set()
    patched: set[int] = set()
    frozen: set[int] = set()
    message_hashes: dict[int, str] = {}
    previous_captured_at = 0.0
    finished_page: int | None = None

    for expected_sequence, raw_event in enumerate(raw_events, start=1):
        if not isinstance(raw_event, dict) or set(raw_event) != _EVENT_FIELDS:
            raise ValueError("invalid live event fields")
        sequence = raw_event["sequence"]
        page = raw_event["page"]
        kind = raw_event["kind"]
        captured_at = raw_event["captured_at"]
        if isinstance(sequence, bool) or sequence != expected_sequence:
            raise ValueError("live event sequences must be contiguous and ordered")
        if isinstance(page, bool) or not isinstance(page, int) or page < 0:
            raise ValueError("live event page must be a non-negative integer")
        if kind not in _EVENT_KINDS:
            raise ValueError(f"unsupported live event kind: {kind}")
        if (
            isinstance(captured_at, bool)
            or not isinstance(captured_at, (int, float))
            or captured_at <= 0
            or captured_at < previous_captured_at
        ):
            raise ValueError("live event timestamps must be positive and monotonic")
        previous_captured_at = float(captured_at)
        message_hash = _require_hash(raw_event["message_id_hash"], "message_id_hash")
        _require_hash(raw_event["payload_sha256"], "payload_sha256")

        if finished_page is not None:
            raise ValueError("finish must be the final live event")
        existing_hash = message_hashes.get(page)
        if existing_hash is not None and existing_hash != message_hash:
            raise ValueError(f"page {page} changed message identity")

        if kind == "create":
            if page != len(created) or page in created:
                raise ValueError("pages must be created once in contiguous order")
            created.add(page)
            message_hashes[page] = message_hash
        elif page not in created:
            raise ValueError(f"page {page} was mutated before create")
        elif kind == "patch":
            if page in frozen:
                raise ValueError(f"frozen page {page} was patched")
            patched.add(page)
        elif kind == "freeze":
            if page in frozen:
                raise ValueError(f"page {page} was frozen more than once")
            if page + 1 not in created:
                raise ValueError(f"page {page + 1} must be visible before page {page} is frozen")
            frozen.add(page)
        elif kind == "finish":
            if page in frozen or page != max(created):
                raise ValueError("only the newest visible page may finish")
            finished_page = page

    if len(created) < 2:
        raise ValueError("capacity continuation requires at least two created pages")
    newest_page = max(created)
    if patched != created:
        raise ValueError("every created page must have an observed PATCH")
    if frozen != set(range(newest_page)):
        raise ValueError("every historical page must be frozen exactly once")
    if finished_page != newest_page:
        raise ValueError("the newest page must have a terminal finish event")
    return len(raw_events), len(created)


def _evaluate_capture(
    path: Path,
    binding: WorkflowTenantBinding,
) -> tuple[int, int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    required_fields = {
        "schema_version",
        "binding",
        "events",
        "checks",
        "artifacts",
        "observed_result_count",
        "attestor",
    }
    if not isinstance(raw, dict) or set(raw) != required_fields:
        raise ValueError("live results must be a bound Workflow capture envelope")
    if raw["schema_version"] != 1:
        raise ValueError("unsupported live capture schema")
    if raw["binding"] != asdict(binding):
        raise ValueError("live capture binding does not match requested run")

    checks = raw["checks"]
    if not isinstance(checks, dict) or set(checks) != _CHECK_NAMES:
        raise ValueError("live capture checks do not match the required checklist")
    incomplete_checks = sorted(name for name, value in checks.items() if value is not True)
    if incomplete_checks:
        raise ValueError("live capture has incomplete checks: " + ", ".join(incomplete_checks))

    artifacts = raw["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != _ARTIFACT_NAMES:
        raise ValueError("live capture artifacts do not match the required checklist")
    for name, digest in artifacts.items():
        _require_hash(digest, name)

    observed_result_count = raw["observed_result_count"]
    if (
        isinstance(observed_result_count, bool)
        or observed_result_count != binding.expected_result_count
    ):
        raise ValueError("observed result count does not match the bound expectation")
    attestor = raw["attestor"]
    if not isinstance(attestor, str) or not attestor.strip() or len(attestor) > 256:
        raise ValueError("live capture requires a non-empty attestor")

    return _validate_events(raw["events"])


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        binding, missing = _binding(args)
        if binding is None:
            _emit(
                status="pending",
                live_mode=args.live,
                reason="missing non-secret Workflow run binding",
                missing_environment=missing,
            )
            return 2

        if args.template_out is not None:
            if args.live or args.live_results is not None:
                raise ValueError("--template-out cannot be combined with live evaluation")
            _write_template(args.template_out, binding)
            _emit(
                status="template_created",
                live_mode=False,
                path=str(args.template_out),
                run_id=binding.run_id,
            )
            return 0

        if not args.live:
            if args.live_results is not None:
                raise ValueError("--live-results requires --live")
            _emit(
                status="pending",
                live_mode=False,
                reason="no real-tenant capture evaluated",
                run_id=binding.run_id,
            )
            return 2
        if os.environ.get("GHOSTAP_WORKFLOW_ACCEPTANCE_LIVE") != "1":
            _emit(
                status="failed",
                live_mode=True,
                reason="live mode requires GHOSTAP_WORKFLOW_ACCEPTANCE_LIVE=1",
            )
            return 1
        if args.live_results is None:
            raise ValueError("live mode requires --live-results")

        event_count, page_count = _evaluate_capture(args.live_results, binding)
        _emit(
            status="passed",
            live_mode=True,
            run_id=binding.run_id,
            event_count=event_count,
            page_count=page_count,
        )
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _emit(status="failed", live_mode=args.live, reason=str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
