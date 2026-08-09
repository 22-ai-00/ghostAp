"""Tests for ControlAuthorizationGate: cross-user, owner, admin, default-deny."""

import time

import pytest

from src.autonomous.domain.control import Principal
from src.autonomous.policy.authorization import (
    AuthorizationDenied,
    AuthorizationEnvelope,
    ControlAuthorizationGate,
    Operation,
    ResourceACL,
)


def _make_principal(
    principal_id: str = "p_user1",
    tenant_key: str = "tenant_a",
    user_id: str = "u1",
    roles: tuple[str, ...] = (),
) -> Principal:
    return Principal(
        principal_id=principal_id,
        tenant_key=tenant_key,
        user_id=user_id,
        roles=roles,
    )


def _make_gate(
    admin_ids: tuple[str, ...] = ("p_admin",),
) -> ControlAuthorizationGate:
    return ControlAuthorizationGate(admin_principal_ids=admin_ids)


class TestDefaultDeny:
    """Authorization must default to deny when no ACL is registered."""

    def test_no_acl_raises_denied(self) -> None:
        gate = _make_gate()
        principal = _make_principal()

        with pytest.raises(AuthorizationDenied) as exc_info:
            gate.authorize(principal, Operation.READ, "resource_1")

        assert exc_info.value.reason == "no ACL registered"
        assert exc_info.value.principal_id == "p_user1"
        assert exc_info.value.operation == Operation.READ

    def test_unregistered_resource_denied(self) -> None:
        gate = _make_gate()
        gate.register_acl(
            ResourceACL(
                resource="resource_x",
                owner_principal_id="p_owner",
                tenant_key="tenant_a",
            )
        )
        principal = _make_principal()

        with pytest.raises(AuthorizationDenied):
            gate.authorize(principal, Operation.READ, "resource_y")

    def test_non_admin_non_owner_no_grant_denied(self) -> None:
        gate = _make_gate()
        gate.register_acl(
            ResourceACL(
                resource="res1",
                owner_principal_id="p_owner",
                tenant_key="tenant_a",
            )
        )
        principal = _make_principal(principal_id="p_stranger")

        with pytest.raises(AuthorizationDenied) as exc_info:
            gate.authorize(principal, Operation.EXECUTE, "res1")

        assert exc_info.value.reason == "default deny"


class TestAdminAccess:
    """Admin access derived from settings, not callback payload."""

    def test_admin_by_principal_id(self) -> None:
        gate = _make_gate(admin_ids=("p_admin",))
        admin = _make_principal(principal_id="p_admin")

        result = gate.authorize(admin, Operation.DELETE, "any_resource")
        assert result.allowed is True
        assert result.reason == "admin"

    def test_admin_by_role(self) -> None:
        gate = ControlAuthorizationGate(
            admin_principal_ids=(), admin_roles=("superuser",)
        )
        admin = _make_principal(principal_id="p_someone", roles=("superuser",))

        result = gate.authorize(admin, Operation.DELETE, "any_resource")
        assert result.allowed is True
        assert result.reason == "admin"

    def test_is_admin_never_from_callback_data(self) -> None:
        """is_admin uses only configured IDs and roles."""
        gate = _make_gate(admin_ids=("p_real_admin",))
        # Even if we pass a principal that claims admin role,
        # it must match configured roles
        fake = _make_principal(
            principal_id="p_imposter", roles=("viewer",)
        )
        assert gate.is_admin(fake) is False

    def test_admin_bypasses_tenant_check(self) -> None:
        gate = _make_gate(admin_ids=("p_admin",))
        gate.register_acl(
            ResourceACL(
                resource="res1",
                owner_principal_id="p_owner",
                tenant_key="tenant_b",
            )
        )
        admin = _make_principal(
            principal_id="p_admin", tenant_key="tenant_a"
        )
        # Admin should bypass even tenant mismatch
        result = gate.authorize(admin, Operation.READ, "res1")
        assert result.allowed is True


class TestOwnerAccess:
    """Resource owner always has full access within same tenant."""

    def test_owner_can_read(self) -> None:
        gate = _make_gate()
        gate.register_acl(
            ResourceACL(
                resource="res1",
                owner_principal_id="p_owner",
                tenant_key="tenant_a",
            )
        )
        owner = _make_principal(principal_id="p_owner", tenant_key="tenant_a")

        result = gate.authorize(owner, Operation.READ, "res1")
        assert result.allowed is True
        assert result.reason == "owner"

    def test_owner_can_execute(self) -> None:
        gate = _make_gate()
        gate.register_acl(
            ResourceACL(
                resource="res1",
                owner_principal_id="p_owner",
                tenant_key="tenant_a",
            )
        )
        owner = _make_principal(principal_id="p_owner", tenant_key="tenant_a")

        result = gate.authorize(owner, Operation.EXECUTE, "res1")
        assert result.allowed is True


class TestCrossUserIsolation:
    """Cross-tenant access is always denied. Cross-user within tenant follows ACL."""

    def test_cross_tenant_denied(self) -> None:
        gate = _make_gate()
        gate.register_acl(
            ResourceACL(
                resource="res1",
                owner_principal_id="p_owner",
                tenant_key="tenant_a",
            )
        )
        cross_tenant = _make_principal(
            principal_id="p_cross", tenant_key="tenant_b"
        )

        with pytest.raises(AuthorizationDenied) as exc_info:
            gate.authorize(cross_tenant, Operation.READ, "res1")

        assert exc_info.value.reason == "tenant mismatch"

    def test_cross_user_same_tenant_needs_grant(self) -> None:
        gate = _make_gate()
        gate.register_acl(
            ResourceACL(
                resource="res1",
                owner_principal_id="p_owner",
                tenant_key="tenant_a",
            )
        )
        other_user = _make_principal(
            principal_id="p_other", tenant_key="tenant_a"
        )

        with pytest.raises(AuthorizationDenied):
            gate.authorize(other_user, Operation.READ, "res1")

    def test_cross_user_with_principal_grant(self) -> None:
        gate = _make_gate()
        gate.register_acl(
            ResourceACL(
                resource="res1",
                owner_principal_id="p_owner",
                tenant_key="tenant_a",
                allowed_principals={"p_collaborator"},
            )
        )
        collaborator = _make_principal(
            principal_id="p_collaborator", tenant_key="tenant_a"
        )

        result = gate.authorize(collaborator, Operation.READ, "res1")
        assert result.allowed is True
        assert result.reason == "principal grant"

    def test_cross_user_with_role_grant(self) -> None:
        gate = _make_gate()
        gate.register_acl(
            ResourceACL(
                resource="res1",
                owner_principal_id="p_owner",
                tenant_key="tenant_a",
                allowed_roles={"reviewer"},
            )
        )
        reviewer = _make_principal(
            principal_id="p_reviewer",
            tenant_key="tenant_a",
            roles=("reviewer",),
        )

        result = gate.authorize(reviewer, Operation.READ, "res1")
        assert result.allowed is True
        assert result.reason == "role grant"

    def test_operation_specific_grant(self) -> None:
        gate = _make_gate()
        gate.register_acl(
            ResourceACL(
                resource="res1",
                owner_principal_id="p_owner",
                tenant_key="tenant_a",
                allowed_principals={"p_viewer"},
                allowed_operations={
                    "p_viewer": {Operation.READ},
                },
            )
        )
        viewer = _make_principal(
            principal_id="p_viewer", tenant_key="tenant_a"
        )

        # READ allowed
        result = gate.authorize(viewer, Operation.READ, "res1")
        assert result.allowed is True

        # EXECUTE not in operation grant
        with pytest.raises(AuthorizationDenied):
            gate.authorize(viewer, Operation.EXECUTE, "res1")


class TestEnvelopeCommit:
    """Authorization envelopes are immutable once committed."""

    def test_commit_envelope_succeeds(self) -> None:
        gate = _make_gate()
        envelope = AuthorizationEnvelope(
            principal_id="p_user1",
            tenant_key="tenant_a",
            operation="execute",
            resource="goal_1",
            expires_at=time.time() + 3600,
        )

        assert gate.commit_envelope(envelope) is True
        stored = gate.get_envelope(envelope.envelope_id)
        assert stored is not None
        assert stored.nonce == envelope.nonce

    def test_expired_envelope_rejected(self) -> None:
        gate = _make_gate()
        envelope = AuthorizationEnvelope(
            principal_id="p_user1",
            tenant_key="tenant_a",
            operation="execute",
            resource="goal_1",
            expires_at=time.time() - 100,  # already expired
        )

        assert gate.commit_envelope(envelope) is False

    def test_envelope_content_hash_stable(self) -> None:
        envelope = AuthorizationEnvelope(
            principal_id="p_user1",
            tenant_key="tenant_a",
            operation="execute",
            resource="goal_1",
        )
        hash1 = envelope.content_hash
        hash2 = envelope.content_hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex


class TestDependentAuthorization:
    """Dependent authorization creation within the same logical frame."""

    def test_create_dependent_from_committed_parent(self) -> None:
        gate = _make_gate()
        parent = AuthorizationEnvelope(
            principal_id="p_user1",
            tenant_key="tenant_a",
            operation="activate",
            resource="goal_1",
            expires_at=time.time() + 3600,
        )
        gate.commit_envelope(parent)

        principal = _make_principal(principal_id="p_user1")
        child = gate.create_dependent_authorization(
            parent,
            principal,
            Operation.EXECUTE,
            "step_1",
            budget_id="budget_1",
            budget_amount=100.0,
        )

        assert child.version == parent.version + 1
        assert child.epoch == parent.epoch
        assert gate.get_envelope(child.envelope_id) is not None

    def test_dependent_from_uncommitted_parent_fails(self) -> None:
        gate = _make_gate()
        parent = AuthorizationEnvelope(
            principal_id="p_user1",
            tenant_key="tenant_a",
            operation="activate",
            resource="goal_1",
        )
        # Not committed

        principal = _make_principal(principal_id="p_user1")
        with pytest.raises(AuthorizationDenied) as exc_info:
            gate.create_dependent_authorization(
                parent, principal, Operation.EXECUTE, "step_1"
            )
        assert "parent envelope not committed" in exc_info.value.reason
