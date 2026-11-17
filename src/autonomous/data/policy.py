"""Exact authenticated label policy for employee data blobs."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Mapping

from ..journal.blob_store import BlobRef

_LABEL_FIELDS = frozenset(
    {"tenant_key", "owner_principal_id", "classification", "purpose", "resource_id", "schema_version"}
)
_PURPOSES = frozenset(
    {
        "execution_history",
        "l1_memory",
        "memory_summary",
    }
)
_SECRET_NAME = re.compile(r"(?i)(secret|token|password|credential|api[_-]?key)")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} is required")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{name} contains control characters")
    return value


def _labels(
    *,
    tenant_key: str,
    owner_principal_id: str,
    purpose: str,
    resource_id: str,
) -> dict[str, str]:
    if purpose not in _PURPOSES:
        raise ValueError("invalid data purpose")
    return {
        "tenant_key": _require_text(tenant_key, "tenant_key"),
        "owner_principal_id": _require_text(owner_principal_id, "owner_principal_id"),
        "classification": "restricted",
        "purpose": purpose,
        "resource_id": _require_text(resource_id, "resource_id"),
        "schema_version": "1",
    }


def build_history_labels(
    tenant_key: str,
    owner_principal_id: str,
    record_id: str,
) -> dict[str, str]:
    if re.fullmatch(r"hist_[0-9a-f]{64}", record_id) is None:
        raise ValueError("record_id must be canonical")
    return _labels(
        tenant_key=tenant_key,
        owner_principal_id=owner_principal_id,
        purpose="execution_history",
        resource_id=record_id,
    )


def build_document_labels(
    *,
    tenant_key: str,
    owner_principal_id: str,
    document_id: str,
    kind: str,
) -> dict[str, str]:
    if kind == "execution_history" or kind not in _PURPOSES:
        raise ValueError("invalid document kind")
    if re.fullmatch(r"data_[0-9a-f]{16}", document_id) is None:
        raise ValueError("document_id must be canonical")
    return _labels(
        tenant_key=tenant_key,
        owner_principal_id=owner_principal_id,
        purpose=kind,
        resource_id=document_id,
    )


def validate_blob_ref_labels(ref: BlobRef, expected: Mapping[str, str]) -> None:
    if not isinstance(ref, BlobRef):
        raise ValueError("malformed BlobRef")
    if not isinstance(expected, Mapping) or set(expected) != _LABEL_FIELDS:
        raise ValueError("invalid expected labels")
    if any(
        not isinstance(key, str)
        or not isinstance(value, str)
        or _SECRET_NAME.search(key)
        for key, value in expected.items()
    ):
        raise ValueError("invalid or secret-like labels")
    canonical = _labels(
        tenant_key=expected["tenant_key"],
        owner_principal_id=expected["owner_principal_id"],
        purpose=expected["purpose"],
        resource_id=expected["resource_id"],
    )
    resource_id = canonical["resource_id"]
    if canonical["purpose"] == "execution_history":
        if re.fullmatch(r"hist_[0-9a-f]{64}", resource_id) is None:
            raise ValueError("resource_id does not match history purpose")
    elif re.fullmatch(r"data_[0-9a-f]{16}", resource_id) is None:
        raise ValueError("resource_id does not match document purpose")
    if dict(expected) != canonical or dict(ref.labels or {}) != canonical:
        raise ValueError("BlobRef labels do not match metadata")
    for name in ("blob_hash", "payload_hash", "labels_hash"):
        if _SHA256_RE.fullmatch(getattr(ref, name)) is None:
            raise ValueError("malformed BlobRef hash")
    if not isinstance(ref.key_ref, str) or not ref.key_ref:
        raise ValueError("malformed BlobRef key_ref")
    if isinstance(ref.size, bool) or not isinstance(ref.size, int) or ref.size < 0:
        raise ValueError("malformed BlobRef size")
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(encoded).hexdigest() != ref.labels_hash:
        raise ValueError("BlobRef labels hash mismatch")
