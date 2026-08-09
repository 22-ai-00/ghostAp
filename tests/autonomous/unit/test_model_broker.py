"""Tests for ModelBroker: budget, rate-limiting, hash binding, state machine."""

import time

import pytest

from src.autonomous.broker.model_broker import (
    ModelBroker,
    ModelCallState,
    RateLimiter,
    _compute_hash,
)
from src.autonomous.domain import GoalActivationAuthorization
from src.autonomous.policy.budget_manager import BudgetManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def budget_manager() -> BudgetManager:
    mgr = BudgetManager()
    mgr.get_or_create_ledger(
        run_id="run_1",
        goal_id="goal_1",
        limits={"model_cost": 100.0},
    )
    return mgr


@pytest.fixture
def ledger_id(budget_manager: BudgetManager) -> str:
    """Return the ledger_id of the first created ledger."""
    for lid in budget_manager._ledgers:
        return lid
    raise RuntimeError("no ledger created")


@pytest.fixture
def authorization() -> GoalActivationAuthorization:
    return GoalActivationAuthorization(
        goal_id="goal_1",
        expires_at=time.time() + 3600,
    )


@pytest.fixture
def expired_auth() -> GoalActivationAuthorization:
    return GoalActivationAuthorization(
        goal_id="goal_1",
        expires_at=time.time() - 100,
    )


async def mock_model_fn(prompt: dict) -> dict:
    """Simple mock model that echoes the prompt."""
    return {"output_type": "submit_output", "content": {"echo": prompt}}


async def failing_model_fn(prompt: dict) -> dict:
    """Model that always fails."""
    raise RuntimeError("Model unavailable")


@pytest.fixture
def broker(budget_manager: BudgetManager, ledger_id: str) -> ModelBroker:
    return ModelBroker(
        model_fn=mock_model_fn,
        budget_manager=budget_manager,
        ledger_id=ledger_id,
        default_model_id="test-model",
        cost_per_call=1.0,
    )


# ---------------------------------------------------------------------------
# State machine tests
# ---------------------------------------------------------------------------


class TestModelCallStateMachine:
    """ModelCall transitions: REQUESTED -> EXECUTING -> COMMITTED/FAILED."""

    @pytest.mark.asyncio
    async def test_successful_call_transitions_to_committed(
        self, broker: ModelBroker, authorization: GoalActivationAuthorization
    ) -> None:
        result = await broker.call(
            authorization=authorization,
            prompt_ref={"goal": "test"},
            run_id="run_1",
            attempt_id="att_1",
        )
        assert result.success is True
        call = broker.get_call(result.call_id)
        assert call is not None
        assert call.state == ModelCallState.COMMITTED

    @pytest.mark.asyncio
    async def test_failed_call_transitions_to_failed(
        self, budget_manager: BudgetManager, ledger_id: str, authorization: GoalActivationAuthorization
    ) -> None:
        broker = ModelBroker(
            model_fn=failing_model_fn,
            budget_manager=budget_manager,
            ledger_id=ledger_id,
            default_model_id="test-model",
        )
        result = await broker.call(
            authorization=authorization,
            prompt_ref={"goal": "test"},
        )
        assert result.success is False
        assert "unavailable" in result.error
        call = broker.get_call(result.call_id)
        assert call is not None
        assert call.state == ModelCallState.FAILED


# ---------------------------------------------------------------------------
# Budget tests
# ---------------------------------------------------------------------------


class TestBudgetEnforcement:
    """Budget is reserved before call and settled/released after."""

    @pytest.mark.asyncio
    async def test_budget_reserved_before_call(
        self, broker: ModelBroker, authorization: GoalActivationAuthorization,
        budget_manager: BudgetManager, ledger_id: str,
    ) -> None:
        ledger = budget_manager.get_ledger(ledger_id)
        initial = ledger.available("model_cost")
        result = await broker.call(
            authorization=authorization,
            prompt_ref={"test": 1},
            run_id="run_1",
        )
        assert result.success is True
        # Budget should be consumed (settled)
        ledger_after = budget_manager.get_ledger(ledger_id)
        assert ledger_after.available("model_cost") < initial

    @pytest.mark.asyncio
    async def test_budget_exhausted_rejects_call(
        self, authorization: GoalActivationAuthorization
    ) -> None:
        # Create a ledger with very low budget
        mgr = BudgetManager()
        ledger = mgr.get_or_create_ledger(
            run_id="run_1", goal_id="goal_1", limits={"model_cost": 0.5}
        )
        broker = ModelBroker(
            model_fn=mock_model_fn,
            budget_manager=mgr,
            ledger_id=ledger.ledger_id,
            cost_per_call=1.0,
        )
        result = await broker.call(
            authorization=authorization,
            prompt_ref={"test": 1},
        )
        assert result.success is False
        assert "Budget" in result.error

    @pytest.mark.asyncio
    async def test_budget_released_on_failure(
        self, authorization: GoalActivationAuthorization
    ) -> None:
        mgr = BudgetManager()
        ledger = mgr.get_or_create_ledger(
            run_id="run_1", goal_id="goal_1", limits={"model_cost": 10.0}
        )
        broker = ModelBroker(
            model_fn=failing_model_fn,
            budget_manager=mgr,
            ledger_id=ledger.ledger_id,
            cost_per_call=1.0,
        )
        initial_available = ledger.available("model_cost")
        result = await broker.call(
            authorization=authorization,
            prompt_ref={"test": 1},
        )
        assert result.success is False
        # Budget should be released (back to available)
        ledger_after = mgr.get_ledger(ledger.ledger_id)
        assert ledger_after.available("model_cost") == initial_available


# ---------------------------------------------------------------------------
# Rate limiting tests
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Rate limiter prevents excessive model calls."""

    def test_rate_limiter_allows_within_limit(self) -> None:
        limiter = RateLimiter(max_calls_per_minute=5, max_calls_per_hour=100)
        for _ in range(5):
            assert limiter.check() is True
            limiter.record()

    def test_rate_limiter_blocks_over_minute_limit(self) -> None:
        limiter = RateLimiter(max_calls_per_minute=2, max_calls_per_hour=100)
        limiter.record()
        limiter.record()
        assert limiter.check() is False

    @pytest.mark.asyncio
    async def test_rate_limited_call_rejected(
        self, budget_manager: BudgetManager, ledger_id: str, authorization: GoalActivationAuthorization
    ) -> None:
        limiter = RateLimiter(max_calls_per_minute=1, max_calls_per_hour=10)
        limiter.record()  # Exhaust the limit

        broker = ModelBroker(
            model_fn=mock_model_fn,
            budget_manager=budget_manager,
            ledger_id=ledger_id,
            rate_limiter=limiter,
        )
        result = await broker.call(
            authorization=authorization,
            prompt_ref={"test": 1},
        )
        assert result.success is False
        assert "Rate" in result.error


# ---------------------------------------------------------------------------
# Authorization tests
# ---------------------------------------------------------------------------


class TestAuthorizationValidation:
    """Authorization is checked before every call."""

    @pytest.mark.asyncio
    async def test_expired_auth_rejected(
        self, broker: ModelBroker, expired_auth: GoalActivationAuthorization
    ) -> None:
        result = await broker.call(
            authorization=expired_auth,
            prompt_ref={"test": 1},
        )
        assert result.success is False
        assert "expired" in result.error.lower() or "consumed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_consumed_auth_rejected(
        self, broker: ModelBroker
    ) -> None:
        auth = GoalActivationAuthorization(
            goal_id="goal_1",
            expires_at=time.time() + 3600,
        )
        consumed_auth = auth.consume()
        result = await broker.call(
            authorization=consumed_auth,
            prompt_ref={"test": 1},
        )
        assert result.success is False


# ---------------------------------------------------------------------------
# Hash binding tests
# ---------------------------------------------------------------------------


class TestHashBinding:
    """Prompt and response hashes are computed and stored."""

    @pytest.mark.asyncio
    async def test_prompt_hash_in_result(
        self, broker: ModelBroker, authorization: GoalActivationAuthorization
    ) -> None:
        prompt = {"goal": "hash_test", "data": [1, 2, 3]}
        result = await broker.call(
            authorization=authorization,
            prompt_ref=prompt,
        )
        assert result.success is True
        assert result.prompt_hash == _compute_hash(prompt)

    @pytest.mark.asyncio
    async def test_response_hash_in_result(
        self, broker: ModelBroker, authorization: GoalActivationAuthorization
    ) -> None:
        result = await broker.call(
            authorization=authorization,
            prompt_ref={"goal": "test"},
        )
        assert result.success is True
        assert result.response_hash != ""
        assert len(result.response_hash) == 64  # SHA256 hex

    @pytest.mark.asyncio
    async def test_same_prompt_same_hash(
        self, broker: ModelBroker, authorization: GoalActivationAuthorization
    ) -> None:
        prompt = {"key": "value"}
        r1 = await broker.call(authorization=authorization, prompt_ref=prompt)
        r2 = await broker.call(authorization=authorization, prompt_ref=prompt)
        assert r1.prompt_hash == r2.prompt_hash


# ---------------------------------------------------------------------------
# Ledger queries
# ---------------------------------------------------------------------------


class TestLedgerQueries:
    """Broker tracks all calls in the ledger."""

    @pytest.mark.asyncio
    async def test_get_call_count(
        self, broker: ModelBroker, authorization: GoalActivationAuthorization
    ) -> None:
        await broker.call(authorization=authorization, prompt_ref={"n": 1}, run_id="r1")
        await broker.call(authorization=authorization, prompt_ref={"n": 2}, run_id="r1")
        assert broker.get_call_count("r1") == 2
        assert broker.get_call_count("r2") == 0

    @pytest.mark.asyncio
    async def test_get_total_cost(
        self, broker: ModelBroker, authorization: GoalActivationAuthorization
    ) -> None:
        await broker.call(authorization=authorization, prompt_ref={"n": 1}, run_id="r1")
        await broker.call(authorization=authorization, prompt_ref={"n": 2}, run_id="r1")
        assert broker.get_total_cost("r1") == 2.0

    @pytest.mark.asyncio
    async def test_model_id_override(
        self, broker: ModelBroker, authorization: GoalActivationAuthorization
    ) -> None:
        result = await broker.call(
            authorization=authorization,
            prompt_ref={"test": 1},
            model_id="custom-model",
        )
        call = broker.get_call(result.call_id)
        assert call is not None
        assert call.model_id == "custom-model"
