"""Security tests for approval nonce: one-time use, replay prevention."""

import time
import uuid

import pytest

from src.autonomous.domain.control import GoalActivationAuthorization, Principal
from src.autonomous.policy.authorization import (
    AuthorizationEnvelope,
    ControlAuthorizationGate,
    Operation,
)


@pytest.fixture
def gate() -> ControlAuthorizationGate:
    return ControlAuthorizationGate(admin_principal_ids=("p_admin",))


class TestNonceOneTimeUse:
    """Nonces are consumed on first use and rejected on replay."""

    def test_fresh_nonce_succeeds(self, gate: ControlAuthorizationGate) -> None:
        nonce = uuid.uuid4().hex
        assert gate.consume_nonce(nonce) is True

    def test_same_nonce_replay_fails(
        self, gate: ControlAuthorizationGate
    ) -> None:
        nonce = uuid.uuid4().hex
        assert gate.consume_nonce(nonce) is True
        assert gate.consume_nonce(nonce) is False

    def test_different_nonces_both_succeed(
        self, gate: ControlAuthorizationGate
    ) -> None:
        nonce_a = uuid.uuid4().hex
        nonce_b = uuid.uuid4().hex
        assert gate.consume_nonce(nonce_a) is True
        assert gate.consume_nonce(nonce_b) is True

    def test_many_nonces_all_unique(
        self, gate: ControlAuthorizationGate
    ) -> None:
        nonces = [uuid.uuid4().hex for _ in range(100)]
        for nonce in nonces:
            assert gate.consume_nonce(nonce) is True
        # All should now be consumed
        for nonce in nonces:
            assert gate.consume_nonce(nonce) is False


class TestEnvelopeNonceReplay:
    """Envelope commit uses nonce to prevent replay attacks."""

    def test_envelope_commit_consumes_nonce(
        self, gate: ControlAuthorizationGate
    ) -> None:
        envelope = AuthorizationEnvelope(
            principal_id="p_user1",
            tenant_key="tenant_a",
            operation="execute",
            resource="goal_1",
            expires_at=time.time() + 3600,
        )
        assert gate.commit_envelope(envelope) is True
        # Same envelope (same nonce) cannot be committed again
        assert gate.commit_envelope(envelope) is False

    def test_forged_envelope_same_nonce_rejected(
        self, gate: ControlAuthorizationGate
    ) -> None:
        """Attacker cannot forge a new envelope with a stolen nonce."""
        original = AuthorizationEnvelope(
            principal_id="p_user1",
            tenant_key="tenant_a",
            operation="execute",
            resource="goal_1",
            nonce="stolen_nonce_123",
            expires_at=time.time() + 3600,
        )
        assert gate.commit_envelope(original) is True

        forged = AuthorizationEnvelope(
            principal_id="p_attacker",
            tenant_key="tenant_a",
            operation="delete",
            resource="goal_1",
            nonce="stolen_nonce_123",  # same nonce
            expires_at=time.time() + 3600,
        )
        assert gate.commit_envelope(forged) is False

    def test_expired_envelope_nonce_not_consumed(
        self, gate: ControlAuthorizationGate
    ) -> None:
        """Expired envelopes do not consume their nonce."""
        expired = AuthorizationEnvelope(
            principal_id="p_user1",
            tenant_key="tenant_a",
            operation="execute",
            resource="goal_1",
            nonce="reusable_nonce_456",
            expires_at=time.time() - 100,
        )
        # Expired envelope is rejected
        assert gate.commit_envelope(expired) is False
        # Nonce should still be available (not consumed)
        assert gate.consume_nonce("reusable_nonce_456") is True


class TestGoalActivationAuthorizationNonce:
    """GoalActivationAuthorization nonce is one-time use."""

    def test_activation_consume_invalidates_nonce(self) -> None:
        auth = GoalActivationAuthorization(
            tenant_key="t1",
            principal_id="p1",
            goal_id="g1",
            expires_at=time.time() + 3600,
        )
        assert auth.is_valid() is True
        consumed = auth.consume()
        assert consumed.consumed is True
        assert consumed.consumed_at is not None

        # Re-consuming raises
        with pytest.raises(ValueError, match="expired or consumed"):
            consumed.consume()

    def test_expired_activation_cannot_be_consumed(self) -> None:
        auth = GoalActivationAuthorization(
            tenant_key="t1",
            principal_id="p1",
            goal_id="g1",
            expires_at=time.time() - 100,
        )
        assert auth.is_valid() is False
        with pytest.raises(ValueError, match="expired or consumed"):
            auth.consume()

    def test_activation_nonce_is_unique(self) -> None:
        auth1 = GoalActivationAuthorization(
            tenant_key="t1", principal_id="p1", goal_id="g1"
        )
        auth2 = GoalActivationAuthorization(
            tenant_key="t1", principal_id="p1", goal_id="g1"
        )
        assert auth1.nonce != auth2.nonce


class TestNonceAndDependentCreation:
    """Nonce consumption and dependent authorization in same frame."""

    def test_nonce_consumed_in_dependent_creation(
        self, gate: ControlAuthorizationGate
    ) -> None:
        """Parent nonce consumed, child gets unique nonce atomically."""
        parent = AuthorizationEnvelope(
            principal_id="p_user1",
            tenant_key="tenant_a",
            operation="activate",
            resource="goal_1",
            expires_at=time.time() + 3600,
        )
        gate.commit_envelope(parent)

        principal = Principal(
            principal_id="p_user1",
            tenant_key="tenant_a",
            user_id="u1",
        )
        child = gate.create_dependent_authorization(
            parent, principal, Operation.EXECUTE, "step_1"
        )

        # Parent nonce is consumed
        assert gate.consume_nonce(parent.nonce) is False
        # Child nonce is also consumed (from commit_envelope)
        assert gate.consume_nonce(child.nonce) is False
        # Child is stored
        assert gate.get_envelope(child.envelope_id) is not None

    def test_replay_parent_after_dependent_creation_fails(
        self, gate: ControlAuthorizationGate
    ) -> None:
        """Cannot re-commit parent envelope after dependent was created."""
        parent = AuthorizationEnvelope(
            principal_id="p_user1",
            tenant_key="tenant_a",
            operation="activate",
            resource="goal_1",
            expires_at=time.time() + 3600,
        )
        gate.commit_envelope(parent)

        principal = Principal(
            principal_id="p_user1",
            tenant_key="tenant_a",
            user_id="u1",
        )
        gate.create_dependent_authorization(
            parent, principal, Operation.EXECUTE, "step_1"
        )

        # Attempt to replay parent
        assert gate.commit_envelope(parent) is False

    def test_multiple_dependents_each_get_unique_nonce(
        self, gate: ControlAuthorizationGate
    ) -> None:
        parent = AuthorizationEnvelope(
            principal_id="p_user1",
            tenant_key="tenant_a",
            operation="activate",
            resource="goal_1",
            expires_at=time.time() + 3600,
        )
        gate.commit_envelope(parent)

        principal = Principal(
            principal_id="p_user1",
            tenant_key="tenant_a",
            user_id="u1",
        )
        child1 = gate.create_dependent_authorization(
            parent, principal, Operation.EXECUTE, "step_1"
        )
        child2 = gate.create_dependent_authorization(
            parent, principal, Operation.EXECUTE, "step_2"
        )

        assert child1.nonce != child2.nonce
        assert child1.envelope_id != child2.envelope_id
