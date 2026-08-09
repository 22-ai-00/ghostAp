"""Tests for PolicyEngine: risk matrix, standing orders, authorization types."""


import pytest

from src.autonomous.domain.enums import AutonomyMode, RiskLevel
from src.autonomous.domain.goals import EpochSet
from src.autonomous.policy.kill_switch import KillSwitch
from src.autonomous.policy.policy_engine import (
    AuthorizationType,
    PolicyContext,
    PolicyDecision,
    PolicyEngine,
)


def _make_epochs(kill_epoch: int = 0) -> EpochSet:
    return EpochSet(kill_epoch=kill_epoch)


def _make_context(
    *,
    run_id: str = "run_1",
    step_id: str = "step_1",
    attempt_id: str = "att_1",
    capability: str = "shell_exec",
    risk_level: RiskLevel = RiskLevel.R0,
    autonomy_mode: AutonomyMode = AutonomyMode.SUPERVISED,
    employee_id: str = "emp_1",
    authorization_type: AuthorizationType = AuthorizationType.ACTIVATION,
    model_id: str = "",
    effect_id: str = "",
    parent_auth_id: str = "",
) -> PolicyContext:
    return PolicyContext(
        run_id=run_id,
        step_id=step_id,
        attempt_id=attempt_id,
        capability=capability,
        risk_level=risk_level,
        autonomy_mode=autonomy_mode,
        employee_id=employee_id,
        authorization_type=authorization_type,
        model_id=model_id,
        effect_id=effect_id,
        parent_auth_id=parent_auth_id,
    )


@pytest.fixture
def kill_switch(tmp_path):
    return KillSwitch(str(tmp_path))


@pytest.fixture
def engine(kill_switch):
    return PolicyEngine(kill_switch)


class TestRiskMatrix:
    """Verify the risk x autonomy decision matrix."""

    def test_assist_r0_allows(self, engine: PolicyEngine) -> None:
        ctx = _make_context(
            autonomy_mode=AutonomyMode.ASSIST, risk_level=RiskLevel.R0
        )
        result = engine.evaluate(ctx, _make_epochs())
        assert result.decision == PolicyDecision.ALLOW

    def test_assist_r1_denies(self, engine: PolicyEngine) -> None:
        ctx = _make_context(
            autonomy_mode=AutonomyMode.ASSIST, risk_level=RiskLevel.R1
        )
        result = engine.evaluate(ctx, _make_epochs())
        assert result.decision == PolicyDecision.DENY

    def test_supervised_r2_requires_approval(
        self, engine: PolicyEngine
    ) -> None:
        ctx = _make_context(
            autonomy_mode=AutonomyMode.SUPERVISED, risk_level=RiskLevel.R2
        )
        result = engine.evaluate(ctx, _make_epochs())
        assert result.decision == PolicyDecision.REQUIRE_APPROVAL
        assert result.approval_id is not None

    def test_supervised_r3_denies(self, engine: PolicyEngine) -> None:
        ctx = _make_context(
            autonomy_mode=AutonomyMode.SUPERVISED, risk_level=RiskLevel.R3
        )
        result = engine.evaluate(ctx, _make_epochs())
        assert result.decision == PolicyDecision.DENY

    def test_bounded_autonomous_r2_allows(self, engine: PolicyEngine) -> None:
        ctx = _make_context(
            autonomy_mode=AutonomyMode.BOUNDED_AUTONOMOUS,
            risk_level=RiskLevel.R2,
        )
        result = engine.evaluate(ctx, _make_epochs())
        assert result.decision == PolicyDecision.ALLOW

    def test_bounded_autonomous_r3_requires_approval(
        self, engine: PolicyEngine
    ) -> None:
        ctx = _make_context(
            autonomy_mode=AutonomyMode.BOUNDED_AUTONOMOUS,
            risk_level=RiskLevel.R3,
        )
        result = engine.evaluate(ctx, _make_epochs())
        assert result.decision == PolicyDecision.REQUIRE_APPROVAL

    def test_bounded_autonomous_r4_denies(self, engine: PolicyEngine) -> None:
        ctx = _make_context(
            autonomy_mode=AutonomyMode.BOUNDED_AUTONOMOUS,
            risk_level=RiskLevel.R4,
        )
        result = engine.evaluate(ctx, _make_epochs())
        assert result.decision == PolicyDecision.DENY


class TestKillSwitch:
    """Kill switch integration with policy evaluation."""

    def test_global_kill_denies_all(self, kill_switch, engine) -> None:
        kill_switch.activate("global", "emergency")
        ctx = _make_context(
            autonomy_mode=AutonomyMode.BOUNDED_AUTONOMOUS,
            risk_level=RiskLevel.R0,
        )
        result = engine.evaluate(ctx, _make_epochs(kill_epoch=100))
        assert result.decision == PolicyDecision.DENY
        assert "kill switch active (global)" in result.reasons

    def test_scoped_kill_denies_matching(self, kill_switch, engine) -> None:
        kill_switch.activate("employee:emp_1", "suspended")
        ctx = _make_context(employee_id="emp_1")
        result = engine.evaluate(ctx, _make_epochs())
        assert result.decision == PolicyDecision.DENY

    def test_stale_epoch_denies(self, kill_switch, engine) -> None:
        kill_switch.activate("global", "bump epoch")
        kill_switch.deactivate("global")
        # Epoch is now 1, but context claims epoch 0
        ctx = _make_context()
        result = engine.evaluate(ctx, _make_epochs(kill_epoch=0))
        assert result.decision == PolicyDecision.DENY
        assert "epoch stale" in result.reasons[0]


class TestStandingOrders:
    """Standing orders override default matrix evaluation."""

    def test_standing_order_overrides_matrix(self, engine: PolicyEngine) -> None:
        # Register a standing order that allows shell_exec at R2
        engine.register_standing_order(
            "shell_exec", RiskLevel.R2, PolicyDecision.ALLOW
        )
        ctx = _make_context(
            capability="shell_exec",
            risk_level=RiskLevel.R2,
            autonomy_mode=AutonomyMode.SUPERVISED,
        )
        result = engine.evaluate(ctx, _make_epochs())
        # Matrix would say REQUIRE_APPROVAL, but standing order says ALLOW
        assert result.decision == PolicyDecision.ALLOW
        assert "standing_order" in result.reasons[0]

    def test_no_standing_order_falls_through(
        self, engine: PolicyEngine
    ) -> None:
        ctx = _make_context(
            capability="shell_exec",
            risk_level=RiskLevel.R1,
            autonomy_mode=AutonomyMode.SUPERVISED,
        )
        result = engine.evaluate(ctx, _make_epochs())
        assert result.decision == PolicyDecision.ALLOW


class TestAuthorizationTypes:
    """Type-specific evaluation paths."""

    def test_derived_requires_parent_auth_id(
        self, engine: PolicyEngine
    ) -> None:
        ctx = _make_context(
            authorization_type=AuthorizationType.DERIVED,
            parent_auth_id="",
        )
        result = engine.evaluate(ctx, _make_epochs())
        assert result.decision == PolicyDecision.DENY
        assert "requires parent_auth_id" in result.reasons[0]

    def test_derived_with_parent_evaluates_matrix(
        self, engine: PolicyEngine
    ) -> None:
        ctx = _make_context(
            authorization_type=AuthorizationType.DERIVED,
            parent_auth_id="auth_parent_1",
            risk_level=RiskLevel.R0,
            autonomy_mode=AutonomyMode.SUPERVISED,
        )
        result = engine.evaluate(ctx, _make_epochs())
        assert result.decision == PolicyDecision.ALLOW
        assert result.authorization_type == AuthorizationType.DERIVED

    def test_effect_requires_effect_id(self, engine: PolicyEngine) -> None:
        ctx = _make_context(
            authorization_type=AuthorizationType.EFFECT,
            effect_id="",
        )
        result = engine.evaluate(ctx, _make_epochs())
        assert result.decision == PolicyDecision.DENY

    def test_effect_with_id_evaluates_matrix(
        self, engine: PolicyEngine
    ) -> None:
        ctx = _make_context(
            authorization_type=AuthorizationType.EFFECT,
            effect_id="eff_1",
            risk_level=RiskLevel.R1,
            autonomy_mode=AutonomyMode.SUPERVISED,
        )
        result = engine.evaluate(ctx, _make_epochs())
        assert result.decision == PolicyDecision.ALLOW
        assert result.authorization_type == AuthorizationType.EFFECT

    def test_model_requires_model_id(self, engine: PolicyEngine) -> None:
        ctx = _make_context(
            authorization_type=AuthorizationType.MODEL,
            model_id="",
        )
        result = engine.evaluate(ctx, _make_epochs())
        assert result.decision == PolicyDecision.DENY

    def test_model_with_id_evaluates_matrix(
        self, engine: PolicyEngine
    ) -> None:
        ctx = _make_context(
            authorization_type=AuthorizationType.MODEL,
            model_id="gpt-4",
            risk_level=RiskLevel.R0,
            autonomy_mode=AutonomyMode.SUPERVISED,
        )
        result = engine.evaluate(ctx, _make_epochs())
        assert result.decision == PolicyDecision.ALLOW
        assert result.authorization_type == AuthorizationType.MODEL


class TestApprovals:
    """Approval grant and revoke flow."""

    def test_approval_flow(self, engine: PolicyEngine) -> None:
        ctx = _make_context(
            autonomy_mode=AutonomyMode.SUPERVISED, risk_level=RiskLevel.R2
        )
        result = engine.evaluate(ctx, _make_epochs())
        assert result.decision == PolicyDecision.REQUIRE_APPROVAL
        approval_id = result.approval_id

        # Grant approval
        assert engine.grant_approval(approval_id, "approver_1") is True

        # Re-evaluate same context - should now allow
        result2 = engine.evaluate(ctx, _make_epochs())
        assert result2.decision == PolicyDecision.ALLOW
        assert "pre-approved" in result2.reasons

    def test_revoke_approval(self, engine: PolicyEngine) -> None:
        ctx = _make_context(
            autonomy_mode=AutonomyMode.SUPERVISED, risk_level=RiskLevel.R2
        )
        result = engine.evaluate(ctx, _make_epochs())
        approval_id = result.approval_id

        engine.grant_approval(approval_id, "approver_1")
        engine.revoke_approval(approval_id)

        # After revoke, should require approval again
        result2 = engine.evaluate(ctx, _make_epochs())
        assert result2.decision == PolicyDecision.REQUIRE_APPROVAL

    def test_grant_nonexistent_approval_fails(
        self, engine: PolicyEngine
    ) -> None:
        assert engine.grant_approval("fake_id", "approver_1") is False

    def test_double_grant_fails(self, engine: PolicyEngine) -> None:
        ctx = _make_context(
            autonomy_mode=AutonomyMode.SUPERVISED, risk_level=RiskLevel.R2
        )
        result = engine.evaluate(ctx, _make_epochs())
        approval_id = result.approval_id

        assert engine.grant_approval(approval_id, "approver_1") is True
        assert engine.grant_approval(approval_id, "approver_2") is False
