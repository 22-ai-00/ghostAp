"""Fail-closed Agent Card validation for the outbound A2A pilot.

The pilot deliberately does not use the SDK card resolver or transport
selection.  Registration is the authority for both URLs and credentials;
the public card is untrusted capability data only.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import re
import warnings
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from a2a.types import AgentCard, AgentInterface
from a2a.utils.errors import InvalidParamsError
from a2a.utils.proto_utils import validate_proto_required_fields
from google.protobuf.json_format import ParseDict, ParseError

MAX_AGENT_CARD_BYTES = 64 * 1024
WIRE_PROTOCOL_VERSION = "1.0"
JSONRPC_PROTOCOL_BINDING = "JSONRPC"
SUPPORTED_INPUT_MODE = "text/plain"
SUPPORTED_OUTPUT_MODES = frozenset({"application/json", "text/plain"})
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class AgentCardValidationError(ValueError):
    """A safe, non-reflective Agent Card validation failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"untrusted Agent Card rejected ({code})")


@dataclass(frozen=True, slots=True)
class PilotAgentRegistration:
    """Authoritative, local registration for one admitted pilot agent."""

    tenant_key: str
    agent_id: str
    card_url: str
    endpoint_url: str
    expected_card_digest: str
    credential_ref: str
    remote_tenant: str = ""

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (
                self.tenant_key,
                self.agent_id,
                self.card_url,
                self.endpoint_url,
                self.expected_card_digest,
                self.credential_ref,
            )
        ):
            raise AgentCardValidationError("invalid-registration")
        validate_public_https_url(self.card_url)
        validate_public_https_url(self.endpoint_url)
        if _SHA256_RE.fullmatch(self.expected_card_digest) is None:
            raise AgentCardValidationError("invalid-registration-digest")
        try:
            self.to_remote_descriptor(card_digest=self.expected_card_digest)
        except (TypeError, ValueError):
            raise AgentCardValidationError("invalid-registration") from None

    def to_remote_descriptor(self, *, card_digest: str) -> Any:
        """Validate and build the SDK-free frozen registration binding."""

        from src.autonomous.remote.models import (  # noqa: PLC0415
            RemoteAgentDescriptor,
            RemoteProtocolBinding,
        )

        return RemoteAgentDescriptor(
            tenant_key=self.tenant_key,
            agent_id=self.agent_id,
            card_url=self.card_url,
            endpoint_url=self.endpoint_url,
            card_digest=card_digest,
            credential_ref=self.credential_ref,
            protocol_binding=RemoteProtocolBinding.JSONRPC,
            protocol_version=WIRE_PROTOCOL_VERSION,
            remote_tenant=self.remote_tenant,
        )

    @property
    def expected_canonical_sha256(self) -> str:
        """Explicit alias used by configuration and audit code."""

        return self.expected_card_digest


@dataclass(frozen=True, slots=True)
class TrustedAgentCard:
    """A digest-bound card plus its registration-selected interface."""

    registration: PilotAgentRegistration
    canonical_digest: str
    sdk_card: AgentCard = field(repr=False)
    selected_interface: AgentInterface = field(repr=False)

    def to_remote_descriptor(self) -> Any:
        """Build the SDK-free Phase 0 descriptor without importing it eagerly."""

        return self.registration.to_remote_descriptor(
            card_digest=self.canonical_digest,
        )

    @property
    def accepted_output_modes(self) -> tuple[str, ...]:
        """Return the bounded output modes admitted by the local codec."""

        return tuple(mode for mode in self.sdk_card.default_output_modes if mode in SUPPORTED_OUTPUT_MODES)


def validate_public_https_url(url: str) -> None:
    """Reject URL forms that can trivially cross the pilot trust boundary."""

    if (
        not isinstance(url, str)
        or not url
        or url != url.strip()
        or "\\" in url
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in url)
    ):
        raise AgentCardValidationError("unsafe-url")
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        # Accessing port performs urllib's range and syntax validation.
        parsed.port  # noqa: B018
    except ValueError:
        raise AgentCardValidationError("unsafe-url") from None
    if (
        parsed.scheme.lower() != "https"
        or not parsed.netloc
        or hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or bool(parsed.query)
        or bool(parsed.fragment)
    ):
        raise AgentCardValidationError("unsafe-url")

    normalized_host = hostname.rstrip(".").lower()
    if normalized_host != hostname.lower() or not normalized_host.isascii() or "%" in normalized_host:
        raise AgentCardValidationError("unsafe-url")
    if normalized_host == "localhost" or normalized_host.endswith(".localhost"):
        raise AgentCardValidationError("unsafe-url")
    try:
        literal_ip = ipaddress.ip_address(normalized_host)
    except ValueError:
        # Numeric-looking, non-canonical IPv4 spellings (for example 127.1)
        # are resolved inconsistently by URL stacks, so fail closed.
        if (
            normalized_host.replace(".", "").isdigit()
            or re.fullmatch(r"0x[0-9a-f]+", normalized_host) is not None
            or any(
                not label or re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label) is None
                for label in normalized_host.split(".")
            )
        ):
            raise AgentCardValidationError("unsafe-url") from None
    else:
        if not literal_ip.is_global:
            raise AgentCardValidationError("unsafe-url")


def parse_strict_json(raw: bytes | str, *, max_bytes: int) -> Any:
    """Parse bounded JSON while rejecting duplicate keys and non-finite values."""

    if isinstance(raw, bytes):
        encoded = raw
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            raise AgentCardValidationError("invalid-json") from None
    elif isinstance(raw, str):
        try:
            encoded = raw.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            raise AgentCardValidationError("invalid-json") from None
        text = raw
    else:
        raise AgentCardValidationError("invalid-json")
    if len(encoded) > max_bytes:
        raise AgentCardValidationError("card-too-large")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise AgentCardValidationError("duplicate-json-key")
            value[key] = item
        return value

    def reject_constant(_value: str) -> None:
        raise AgentCardValidationError("invalid-json-number")

    try:
        return json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except AgentCardValidationError:
        raise
    except (RecursionError, ValueError, TypeError, json.JSONDecodeError):
        raise AgentCardValidationError("invalid-json") from None


def canonical_json_bytes(value: Any) -> bytes:
    """Return the pilot's deterministic UTF-8 JSON representation."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8", errors="strict")
    except (RecursionError, TypeError, ValueError, UnicodeEncodeError):
        raise AgentCardValidationError("invalid-json") from None


def canonical_card_digest(raw: bytes | str) -> str:
    """Hash strict, canonical card JSON rather than wire whitespace/order."""

    value = parse_strict_json(raw, max_bytes=MAX_AGENT_CARD_BYTES)
    if not isinstance(value, dict):
        raise AgentCardValidationError("invalid-card-shape")
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def load_trusted_agent_card(
    registration: PilotAgentRegistration,
    raw: bytes | str,
) -> TrustedAgentCard:
    """Validate untrusted public JSON against an authoritative registration."""

    value = parse_strict_json(raw, max_bytes=MAX_AGENT_CARD_BYTES)
    if not isinstance(value, dict):
        raise AgentCardValidationError("invalid-card-shape")
    canonical = canonical_json_bytes(value)
    digest = hashlib.sha256(canonical).hexdigest()
    if not hmac.compare_digest(digest, registration.expected_card_digest):
        raise AgentCardValidationError("card-digest-mismatch")

    try:
        parsed_card = ParseDict(
            value,
            AgentCard(),
            ignore_unknown_fields=False,
            max_recursion_depth=100,
        )
        # protobuf 6.33 warns on the SDK's use of FieldDescriptor.label;
        # keep strict SDK-required-field validation without leaking that
        # dependency-internal deprecation at every public Card refresh.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                module=r"a2a\.utils\.proto_utils",
            )
            validate_proto_required_fields(parsed_card)
    except (
        InvalidParamsError,
        ParseError,
        RecursionError,
        TypeError,
        ValueError,
    ):
        raise AgentCardValidationError("invalid-card-schema") from None

    selected = next(
        (
            interface
            for interface in parsed_card.supported_interfaces
            if interface.url == registration.endpoint_url
            and interface.protocol_binding == JSONRPC_PROTOCOL_BINDING
            and interface.protocol_version == WIRE_PROTOCOL_VERSION
            and interface.tenant == registration.remote_tenant
        ),
        None,
    )
    if selected is None:
        raise AgentCardValidationError("registered-interface-missing")
    validate_public_https_url(selected.url)
    if not parsed_card.HasField("capabilities") or not parsed_card.capabilities.streaming:
        raise AgentCardValidationError("streaming-required")
    if any(extension.required for extension in parsed_card.capabilities.extensions):
        raise AgentCardValidationError("required-extension-unsupported")
    if SUPPORTED_INPUT_MODE not in parsed_card.default_input_modes:
        raise AgentCardValidationError("input-mode-unsupported")
    if not SUPPORTED_OUTPUT_MODES.intersection(parsed_card.default_output_modes):
        raise AgentCardValidationError("output-mode-unsupported")

    # Return a defensive copy containing only the registration-selected
    # transport, so no later SDK factory can silently prefer a card-supplied
    # endpoint or binding.
    sdk_card = AgentCard()
    sdk_card.CopyFrom(parsed_card)
    del sdk_card.supported_interfaces[:]
    sdk_card.supported_interfaces.add().CopyFrom(selected)
    selected_copy = AgentInterface()
    selected_copy.CopyFrom(selected)
    return TrustedAgentCard(
        registration=registration,
        canonical_digest=digest,
        sdk_card=sdk_card,
        selected_interface=selected_copy,
    )


__all__ = [
    "AgentCardValidationError",
    "JSONRPC_PROTOCOL_BINDING",
    "MAX_AGENT_CARD_BYTES",
    "PilotAgentRegistration",
    "SUPPORTED_INPUT_MODE",
    "SUPPORTED_OUTPUT_MODES",
    "TrustedAgentCard",
    "WIRE_PROTOCOL_VERSION",
    "canonical_card_digest",
    "canonical_json_bytes",
    "load_trusted_agent_card",
    "parse_strict_json",
    "validate_public_https_url",
]
