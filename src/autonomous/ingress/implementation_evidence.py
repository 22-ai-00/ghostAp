"""Strict local implementation evidence for employee durable ingress.

This evidence is development-only. It is intentionally separate from the
real-tenant employee release manifest and cannot authorize visible employees.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .sdk_capability import (
    CAPABILITY_PROFILE_ID,
    LOCKED_LARK_CHANNEL_WHEEL_SHA256,
)

PHASE3_IMPLEMENTATION_MANIFEST_PATH = Path(__file__).with_name(
    "phase3_implementation_evidence_manifest.json"
)

_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}\Z")
_OUTCOMES = frozenset({"passed", "failed", "skipped"})
_SELECTOR_STATES = frozenset({"collectable", "pending"})
_GATE_FIELDS = frozenset(
    {
        "id",
        "selector",
        "evidence_level",
        "environment",
        "artifact_kind",
        "artifact_profile_id",
        "sdk_wheel_sha256",
        "selector_state",
        "status",
    }
)
_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "gate_id",
        "selector",
        "commit_sha",
        "artifact_kind",
        "artifact_profile_id",
        "artifact_sha256",
        "sdk_wheel_sha256",
        "sdk_capability_artifact_sha256",
        "collected_nodeids",
        "pytest_exit_code",
        "result_summary",
        "result_summary_sha256",
    }
)
_SUMMARY_FIELDS = frozenset({"nodeid", "setup", "call", "teardown"})


class Phase3EvidenceStatus(str, Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON field: {key}")
        value[key] = item
    return value


def _exact_dict(value: Any, fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    missing = fields - set(value)
    extra = set(value) - fields
    if missing:
        raise ValueError(f"{name} missing fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"{name} has unknown fields: {sorted(extra)}")
    return value


def _strict_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 1024 or "\x00" in value:
        raise ValueError(f"invalid {name}")
    return value


def _sha256(value: Any, name: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"invalid {name}")
    return value


@dataclass(frozen=True, slots=True)
class Phase3ImplementationGate:
    id: str
    selector: str
    evidence_level: str
    environment: str
    artifact_kind: str
    artifact_profile_id: str
    sdk_wheel_sha256: str | None
    selector_state: str
    status: str

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or _ID_RE.fullmatch(self.id) is None:
            raise ValueError("invalid Phase 3 evidence id")
        selector = _strict_text(self.selector, "selector")
        if "::" not in selector or not selector.startswith("tests/autonomous/"):
            raise ValueError(f"invalid selector for {self.id}")
        expected_evidence_level = (
            "integration"
            if self.artifact_kind
            in {
                "employee_channel_bridge",
                "employee_router_queue",
                "employee_team_gateway",
                "employee_terminal_pipeline",
            }
            else "chaos_security"
        )
        if self.evidence_level != expected_evidence_level:
            raise ValueError(f"invalid evidence_level for {self.id}")
        _strict_text(self.environment, "environment")
        _strict_text(self.artifact_kind, "artifact_kind")
        _strict_text(self.artifact_profile_id, "artifact_profile_id")
        if self.selector_state not in _SELECTOR_STATES:
            raise ValueError(f"invalid selector_state for {self.id}")
        if self.status != "pending":
            raise ValueError(f"implementation manifest status must remain pending for {self.id}")
        wheel = _sha256(self.sdk_wheel_sha256, "sdk_wheel_sha256", optional=True)
        if self.artifact_kind == "employee_channel_sdk_capability":
            if self.artifact_profile_id != CAPABILITY_PROFILE_ID:
                raise ValueError(f"invalid SDK capability profile for {self.id}")
            if wheel != LOCKED_LARK_CHANNEL_WHEEL_SHA256:
                raise ValueError(f"invalid SDK wheel identity for {self.id}")
        elif self.artifact_kind == "employee_channel_bridge":
            if self.artifact_profile_id != "employee-channel-bridge-v1":
                raise ValueError(f"invalid Channel bridge profile for {self.id}")
            if wheel != LOCKED_LARK_CHANNEL_WHEEL_SHA256:
                raise ValueError(f"invalid SDK wheel identity for {self.id}")
        elif self.artifact_kind == "employee_ingress_ipc_harness":
            if wheel is not None or self.artifact_profile_id == CAPABILITY_PROFILE_ID:
                raise ValueError(f"IPC evidence cannot use SDK capability identity for {self.id}")
        elif self.artifact_kind == "employee_router_queue":
            if (
                self.artifact_profile_id != "employee-router-queue-v1"
                or wheel is not None
                or self.environment != "local_process_harness"
            ):
                raise ValueError(f"invalid Router queue profile for {self.id}")
        elif self.artifact_kind == "employee_team_gateway":
            if (
                self.artifact_profile_id != "employee-team-gateway-v1"
                or wheel is not None
                or self.environment != "local_team_harness"
            ):
                raise ValueError(f"invalid employee Team gateway profile for {self.id}")
        elif self.artifact_kind == "employee_terminal_pipeline":
            if (
                self.artifact_profile_id != "employee-terminal-pipeline-v1"
                or wheel is not None
                or self.environment != "local_process_harness"
            ):
                raise ValueError(f"invalid employee terminal profile for {self.id}")
        elif self.artifact_kind == "employee_attempt_recovery":
            if (
                self.artifact_profile_id != "employee-attempt-recovery-v1"
                or wheel is not None
                or self.environment != "local_process_harness"
            ):
                raise ValueError(f"invalid employee recovery profile for {self.id}")
        else:
            raise ValueError(f"unsupported artifact kind for {self.id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "selector": self.selector,
            "evidence_level": self.evidence_level,
            "environment": self.environment,
            "artifact_kind": self.artifact_kind,
            "artifact_profile_id": self.artifact_profile_id,
            "sdk_wheel_sha256": self.sdk_wheel_sha256,
            "selector_state": self.selector_state,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, value: Any) -> Phase3ImplementationGate:
        return cls(**_exact_dict(value, _GATE_FIELDS, "Phase 3 implementation gate"))


@dataclass(frozen=True, slots=True)
class ImplementationEvidenceResult:
    """One exact pytest result bound to source and artifact identities."""

    schema_version: int
    gate_id: str
    selector: str
    commit_sha: str
    artifact_kind: str
    artifact_profile_id: str
    artifact_sha256: str
    sdk_wheel_sha256: str | None
    sdk_capability_artifact_sha256: str | None
    collected_nodeids: tuple[str, ...]
    pytest_exit_code: int
    result_summary: Mapping[str, str]
    result_summary_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported implementation evidence schema_version")
        if not isinstance(self.gate_id, str) or _ID_RE.fullmatch(self.gate_id) is None:
            raise ValueError("invalid gate_id")
        selector = _strict_text(self.selector, "selector")
        if "::" not in selector:
            raise ValueError("invalid selector")
        if not isinstance(self.commit_sha, str) or _COMMIT_RE.fullmatch(self.commit_sha) is None:
            raise ValueError("invalid commit_sha")
        _strict_text(self.artifact_kind, "artifact_kind")
        _strict_text(self.artifact_profile_id, "artifact_profile_id")
        _sha256(self.artifact_sha256, "artifact_sha256")
        _sha256(self.sdk_wheel_sha256, "sdk_wheel_sha256", optional=True)
        _sha256(
            self.sdk_capability_artifact_sha256,
            "sdk_capability_artifact_sha256",
            optional=True,
        )
        collected = tuple(self.collected_nodeids)
        if collected != (selector,):
            raise ValueError("collected_nodeids must contain the exact selector once")
        object.__setattr__(self, "collected_nodeids", collected)
        if isinstance(self.pytest_exit_code, bool) or not isinstance(self.pytest_exit_code, int):
            raise ValueError("invalid pytest_exit_code")
        if not isinstance(self.result_summary, Mapping):
            raise ValueError("implementation evidence result summary must be an object")
        summary_data = _exact_dict(
            dict(self.result_summary),
            _SUMMARY_FIELDS,
            "implementation evidence result summary",
        )
        if summary_data["nodeid"] != selector:
            raise ValueError("result summary nodeid does not bind exact selector")
        for phase in ("setup", "call", "teardown"):
            if summary_data[phase] not in _OUTCOMES:
                raise ValueError(f"invalid result summary {phase}")
        summary = MappingProxyType(summary_data)
        object.__setattr__(self, "result_summary", summary)
        expected_summary_hash = hashlib.sha256(_canonical_json(summary_data)).hexdigest()
        _sha256(self.result_summary_sha256, "result_summary_sha256")
        if self.result_summary_sha256 != expected_summary_hash:
            raise ValueError("result summary hash mismatch")

    @property
    def passed(self) -> bool:
        return self.pytest_exit_code == 0 and all(
            self.result_summary[phase] == "passed"
            for phase in ("setup", "call", "teardown")
        )

    @classmethod
    def create(
        cls,
        *,
        gate_id: str,
        selector: str,
        commit_sha: str,
        artifact_kind: str,
        artifact_profile_id: str,
        artifact_sha256: str,
        sdk_wheel_sha256: str | None,
        sdk_capability_artifact_sha256: str | None,
        collected_nodeids: tuple[str, ...],
        pytest_exit_code: int,
        setup: str,
        call: str,
        teardown: str,
    ) -> ImplementationEvidenceResult:
        summary = {
            "nodeid": selector,
            "setup": setup,
            "call": call,
            "teardown": teardown,
        }
        return cls(
            schema_version=1,
            gate_id=gate_id,
            selector=selector,
            commit_sha=commit_sha,
            artifact_kind=artifact_kind,
            artifact_profile_id=artifact_profile_id,
            artifact_sha256=artifact_sha256,
            sdk_wheel_sha256=sdk_wheel_sha256,
            sdk_capability_artifact_sha256=sdk_capability_artifact_sha256,
            collected_nodeids=collected_nodeids,
            pytest_exit_code=pytest_exit_code,
            result_summary=summary,
            result_summary_sha256=hashlib.sha256(_canonical_json(summary)).hexdigest(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "gate_id": self.gate_id,
            "selector": self.selector,
            "commit_sha": self.commit_sha,
            "artifact_kind": self.artifact_kind,
            "artifact_profile_id": self.artifact_profile_id,
            "artifact_sha256": self.artifact_sha256,
            "sdk_wheel_sha256": self.sdk_wheel_sha256,
            "sdk_capability_artifact_sha256": self.sdk_capability_artifact_sha256,
            "collected_nodeids": list(self.collected_nodeids),
            "pytest_exit_code": self.pytest_exit_code,
            "result_summary": dict(self.result_summary),
            "result_summary_sha256": self.result_summary_sha256,
        }

    @classmethod
    def from_dict(cls, value: Any) -> ImplementationEvidenceResult:
        return cls(**_exact_dict(value, _RESULT_FIELDS, "implementation evidence result"))


@dataclass(frozen=True, slots=True)
class Phase3EvidenceEvaluation:
    status: Phase3EvidenceStatus
    passed: tuple[str, ...]
    pending: tuple[str, ...]
    failed: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Phase3ImplementationManifest:
    """Dedicated local manifest; never a substitute for tenant release evidence."""

    profile_id: str
    gates: tuple[Phase3ImplementationGate, ...]

    @staticmethod
    def canonical_json_bytes(value: Any) -> bytes:
        return _canonical_json(value)

    @classmethod
    def load(cls, path: str | Path) -> Phase3ImplementationManifest:
        try:
            raw = json.loads(
                Path(path).read_text(encoding="utf-8"),
                object_pairs_hook=_strict_object,
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid Phase 3 implementation manifest") from exc
        if not isinstance(raw, dict) or set(raw) != {
            "schema_version",
            "profile_id",
            "gates",
        }:
            raise ValueError("invalid Phase 3 implementation manifest fields")
        if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
            raise ValueError("unsupported Phase 3 implementation manifest schema")
        profile_id = raw["profile_id"]
        if not isinstance(profile_id, str) or _ID_RE.fullmatch(profile_id) is None:
            raise ValueError("invalid Phase 3 implementation profile_id")
        if not isinstance(raw["gates"], list) or not raw["gates"]:
            raise ValueError("Phase 3 implementation manifest requires gates")
        gates: list[Phase3ImplementationGate] = []
        seen: set[str] = set()
        for item in raw["gates"]:
            gate = Phase3ImplementationGate.from_dict(item)
            if gate.id in seen:
                raise ValueError(f"duplicate Phase 3 evidence id: {gate.id}")
            seen.add(gate.id)
            gates.append(gate)
        return cls(profile_id=profile_id, gates=tuple(gates))

    def gate(self, gate_id: str) -> Phase3ImplementationGate:
        for gate in self.gates:
            if gate.id == gate_id:
                return gate
        raise ValueError(f"unknown Phase 3 evidence id: {gate_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "profile_id": self.profile_id,
            "gates": [gate.to_dict() for gate in self.gates],
        }

    def evaluate(
        self,
        results: Sequence[ImplementationEvidenceResult | Mapping[str, Any]],
        *,
        expected_commit_sha: str,
        expected_artifact_sha256: str,
        expected_sdk_capability_artifact_sha256: str,
    ) -> Phase3EvidenceEvaluation:
        if not isinstance(expected_commit_sha, str) or _COMMIT_RE.fullmatch(expected_commit_sha) is None:
            raise ValueError("invalid expected commit identity")
        _sha256(expected_artifact_sha256, "expected artifact identity")
        _sha256(
            expected_sdk_capability_artifact_sha256,
            "expected SDK capability artifact identity",
        )
        parsed: list[ImplementationEvidenceResult] = []
        seen: set[str] = set()
        for item in results:
            result = (
                item
                if isinstance(item, ImplementationEvidenceResult)
                else ImplementationEvidenceResult.from_dict(dict(item))
            )
            if result.gate_id in seen:
                raise ValueError(f"duplicate implementation evidence id: {result.gate_id}")
            seen.add(result.gate_id)
            gate = self.gate(result.gate_id)
            if gate.selector_state == "pending":
                raise ValueError(f"evidence submitted for pending selector: {gate.id}")
            if result.selector != gate.selector:
                raise ValueError(f"selector mismatch for {gate.id}")
            if result.commit_sha != expected_commit_sha:
                raise ValueError(f"commit identity mismatch for {gate.id}")
            if result.artifact_sha256 != expected_artifact_sha256:
                raise ValueError(f"artifact identity mismatch for {gate.id}")
            if (
                result.artifact_kind != gate.artifact_kind
                or result.artifact_profile_id != gate.artifact_profile_id
                or result.sdk_wheel_sha256 != gate.sdk_wheel_sha256
            ):
                raise ValueError(f"artifact profile mismatch for {gate.id}")
            if gate.artifact_kind in {
                "employee_channel_sdk_capability",
                "employee_channel_bridge",
            }:
                if (
                    result.sdk_capability_artifact_sha256
                    != expected_sdk_capability_artifact_sha256
                ):
                    raise ValueError(f"SDK capability artifact mismatch for {gate.id}")
            elif result.sdk_capability_artifact_sha256 is not None:
                raise ValueError(f"local evidence cannot use SDK capability artifact for {gate.id}")
            parsed.append(result)

        by_id = {result.gate_id: result for result in parsed}
        passed: list[str] = []
        pending: list[str] = []
        failed: list[str] = []
        for gate in self.gates:
            result = by_id.get(gate.id)
            if result is None:
                pending.append(gate.id)
            elif result.passed:
                passed.append(gate.id)
            else:
                failed.append(gate.id)
        status = (
            Phase3EvidenceStatus.FAILED
            if failed
            else Phase3EvidenceStatus.PENDING
            if pending
            else Phase3EvidenceStatus.PASSED
        )
        return Phase3EvidenceEvaluation(
            status=status,
            passed=tuple(passed),
            pending=tuple(pending),
            failed=tuple(failed),
        )
