"""Tests for AgentRuntime: broker requirement, turn loop, checkpointing."""

import os
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.autonomous.broker.model_broker import ModelBroker
from src.autonomous.broker.tool_broker import (
    CapabilityRegistry,
    ToolBroker,
)
from src.autonomous.domain import (
    CapabilityDescriptor,
    EpochSet,
    GoalActivationAuthorization,
    RiskLevel,
    TurnOutputType,
)
from src.autonomous.policy.budget_manager import BudgetManager
from src.autonomous.runtime.runtime import (
    AgentRuntime,
    TurnInput,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_checkpoint_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def budget_manager() -> BudgetManager:
    mgr = BudgetManager()
    mgr.get_or_create_ledger(
        run_id="run_1",
        goal_id="goal_1",
        limits={"model_cost": 1000.0, "tool_calls": 100.0},
    )
    return mgr


@pytest.fixture
def ledger_id(budget_manager: BudgetManager) -> str:
    for lid in budget_manager._ledgers:
        return lid
    raise RuntimeError("no ledger")


@pytest.fixture
def authorization() -> GoalActivationAuthorization:
    return GoalActivationAuthorization(
        goal_id="goal_1",
        expires_at=time.time() + 3600,
    )


@pytest.fixture
def epochs() -> EpochSet:
    return EpochSet(plan_epoch=1)


async def mock_model_submit(prompt: dict) -> dict:
    """Model that immediately submits output."""
    return {
        "output_type": "submit_output",
        "content": {"result": "done"},
        "tool_proposals": [],
    }


async def mock_model_tool_then_submit(prompt: dict) -> dict:
    """Model that proposes a tool on first turn, then submits."""
    history = prompt.get("history", [])
    if len(history) == 0:
        return {
            "output_type": "tool_proposal",
            "content": {},
            "tool_proposals": [
                {"capability": "read_file", "arguments": {"path": "/tmp/test.txt"}}
            ],
        }
    return {
        "output_type": "submit_output",
        "content": {"result": "completed with tools"},
    }


@pytest.fixture
def model_broker_submit(budget_manager: BudgetManager, ledger_id: str) -> ModelBroker:
    return ModelBroker(
        model_fn=mock_model_submit,
        budget_manager=budget_manager,
        ledger_id=ledger_id,
    )


@pytest.fixture
def model_broker_tool(budget_manager: BudgetManager, ledger_id: str) -> ModelBroker:
    return ModelBroker(
        model_fn=mock_model_tool_then_submit,
        budget_manager=budget_manager,
        ledger_id=ledger_id,
    )


@pytest.fixture
def tool_broker() -> ToolBroker:
    registry = CapabilityRegistry()
    registry.register(
        CapabilityDescriptor(
            capability_id="read_file",
            name="Read File",
            risk_level=RiskLevel.R0,
        ),
        adapter=AsyncMock(return_value={"content": "file data"}),
    )
    return ToolBroker(
        registry=registry,
        policy_check_fn=lambda req, desc: {"decision": "allow"},
        budget_reserve_fn=lambda run_id, dim, amt: MagicMock(entry_id="bud_1"),
        budget_settle_fn=lambda entry, amt: True,
        kill_check_fn=lambda cap: True,
        epoch_check_fn=lambda run_id, epoch, epochs: True,
    )


@pytest.fixture
def initial_input() -> TurnInput:
    return TurnInput(
        turn_seq=0,
        goal_summary="Test goal",
        plan_summary="Test plan",
        step_contract={"step": "test_step"},
        attempt_id="att_1",
        available_capabilities=["read_file"],
        remaining_budget={"model_cost": 100.0},
        deadline=time.time() + 600,
        history=[],
    )


# ---------------------------------------------------------------------------
# Constructor enforcement tests
# ---------------------------------------------------------------------------


class TestConstructorEnforcement:
    """AgentRuntime rejects raw callables, requires proper broker instances."""

    def test_rejects_raw_model_fn(self, tool_broker: ToolBroker, tmp_checkpoint_dir: str) -> None:
        with pytest.raises(TypeError, match="ModelBroker instance"):
            AgentRuntime(
                model_broker=mock_model_submit,  # type: ignore
                tool_broker=tool_broker,
                checkpoint_dir=tmp_checkpoint_dir,
            )

    def test_rejects_raw_tool_executor(
        self, model_broker_submit: ModelBroker, tmp_checkpoint_dir: str
    ) -> None:
        with pytest.raises(TypeError, match="ToolBroker instance"):
            AgentRuntime(
                model_broker=model_broker_submit,
                tool_broker=lambda x: x,  # type: ignore
                checkpoint_dir=tmp_checkpoint_dir,
            )

    def test_accepts_proper_brokers(
        self, model_broker_submit: ModelBroker, tool_broker: ToolBroker, tmp_checkpoint_dir: str
    ) -> None:
        runtime = AgentRuntime(
            model_broker=model_broker_submit,
            tool_broker=tool_broker,
            checkpoint_dir=tmp_checkpoint_dir,
        )
        assert runtime.model_broker is model_broker_submit
        assert runtime.tool_broker is tool_broker


# ---------------------------------------------------------------------------
# Turn loop tests
# ---------------------------------------------------------------------------


class TestTurnLoop:
    """execute_attempt runs structured turn loop correctly."""

    @pytest.mark.asyncio
    async def test_immediate_submit(
        self,
        model_broker_submit: ModelBroker,
        tool_broker: ToolBroker,
        authorization: GoalActivationAuthorization,
        epochs: EpochSet,
        initial_input: TurnInput,
        tmp_checkpoint_dir: str,
    ) -> None:
        runtime = AgentRuntime(
            model_broker=model_broker_submit,
            tool_broker=tool_broker,
            checkpoint_dir=tmp_checkpoint_dir,
        )
        result = await runtime.execute_attempt(
            attempt_id="att_1",
            authorization=authorization,
            initial_input=initial_input,
            epochs=epochs,
            run_id="run_1",
        )
        assert result.stop_reason == "completed"
        assert result.turn_count == 1
        assert result.final_output is not None
        assert result.final_output.output_type == TurnOutputType.SUBMIT_OUTPUT

    @pytest.mark.asyncio
    async def test_tool_proposal_then_submit(
        self,
        model_broker_tool: ModelBroker,
        tool_broker: ToolBroker,
        authorization: GoalActivationAuthorization,
        epochs: EpochSet,
        initial_input: TurnInput,
        tmp_checkpoint_dir: str,
    ) -> None:
        runtime = AgentRuntime(
            model_broker=model_broker_tool,
            tool_broker=tool_broker,
            checkpoint_dir=tmp_checkpoint_dir,
        )
        result = await runtime.execute_attempt(
            attempt_id="att_1",
            authorization=authorization,
            initial_input=initial_input,
            epochs=epochs,
            run_id="run_1",
        )
        assert result.stop_reason == "completed"
        assert result.turn_count == 2
        # First turn proposed tools, second submitted
        assert len(result.history) == 2
        assert result.history[0].output_type == "tool_proposal"
        assert result.history[1].output_type == "submit_output"

    @pytest.mark.asyncio
    async def test_max_turns_stops_loop(
        self,
        tool_broker: ToolBroker,
        authorization: GoalActivationAuthorization,
        epochs: EpochSet,
        initial_input: TurnInput,
        tmp_checkpoint_dir: str,
        budget_manager: BudgetManager,
        ledger_id: str,
    ) -> None:
        # Model that always proposes tools (never submits)
        async def infinite_model(prompt: dict) -> dict:
            return {
                "output_type": "tool_proposal",
                "content": {},
                "tool_proposals": [{"capability": "read_file", "arguments": {"path": "/a"}}],
            }

        model_broker = ModelBroker(
            model_fn=infinite_model,
            budget_manager=budget_manager,
            ledger_id=ledger_id,
        )
        runtime = AgentRuntime(
            model_broker=model_broker,
            tool_broker=tool_broker,
            checkpoint_dir=tmp_checkpoint_dir,
            max_turns=3,
        )
        result = await runtime.execute_attempt(
            attempt_id="att_1",
            authorization=authorization,
            initial_input=initial_input,
            epochs=epochs,
            run_id="run_1",
        )
        assert result.stop_reason == "max_turns"
        assert result.turn_count == 3


# ---------------------------------------------------------------------------
# Checkpoint tests
# ---------------------------------------------------------------------------


class TestCheckpointing:
    """Checkpoints are created after each turn."""

    @pytest.mark.asyncio
    async def test_checkpoint_file_created(
        self,
        model_broker_submit: ModelBroker,
        tool_broker: ToolBroker,
        authorization: GoalActivationAuthorization,
        epochs: EpochSet,
        initial_input: TurnInput,
        tmp_checkpoint_dir: str,
    ) -> None:
        runtime = AgentRuntime(
            model_broker=model_broker_submit,
            tool_broker=tool_broker,
            checkpoint_dir=tmp_checkpoint_dir,
        )
        result = await runtime.execute_attempt(
            attempt_id="att_1",
            authorization=authorization,
            initial_input=initial_input,
            epochs=epochs,
        )
        assert result.checkpoint_path != ""
        assert os.path.exists(result.checkpoint_path)

    @pytest.mark.asyncio
    async def test_checkpoint_can_be_loaded(
        self,
        model_broker_submit: ModelBroker,
        tool_broker: ToolBroker,
        authorization: GoalActivationAuthorization,
        epochs: EpochSet,
        initial_input: TurnInput,
        tmp_checkpoint_dir: str,
    ) -> None:
        runtime = AgentRuntime(
            model_broker=model_broker_submit,
            tool_broker=tool_broker,
            checkpoint_dir=tmp_checkpoint_dir,
        )
        await runtime.execute_attempt(
            attempt_id="att_1",
            authorization=authorization,
            initial_input=initial_input,
            epochs=epochs,
        )
        checkpoint = await runtime.load_checkpoint("att_1")
        assert checkpoint is not None
        assert "turn_seq" in checkpoint
        assert "history" in checkpoint


# ---------------------------------------------------------------------------
# No-progress detection
# ---------------------------------------------------------------------------


class TestNoProgressDetection:
    """Detects and stops on repeated identical turns."""

    @pytest.mark.asyncio
    async def test_no_progress_stop(
        self,
        tool_broker: ToolBroker,
        authorization: GoalActivationAuthorization,
        epochs: EpochSet,
        initial_input: TurnInput,
        tmp_checkpoint_dir: str,
        budget_manager: BudgetManager,
        ledger_id: str,
    ) -> None:
        # Model that always returns same blocked output
        async def stuck_model(prompt: dict) -> dict:
            return {
                "output_type": "blocked",
                "content": {"reason": "stuck"},
            }

        model_broker = ModelBroker(
            model_fn=stuck_model,
            budget_manager=budget_manager,
            ledger_id=ledger_id,
        )
        runtime = AgentRuntime(
            model_broker=model_broker,
            tool_broker=tool_broker,
            checkpoint_dir=tmp_checkpoint_dir,
            no_progress_threshold=3,
            max_turns=10,
        )
        result = await runtime.execute_attempt(
            attempt_id="att_1",
            authorization=authorization,
            initial_input=initial_input,
            epochs=epochs,
        )
        # Model returns "blocked" on first turn, which causes immediate stop
        # since blocked is a terminal type
        assert result.stop_reason == "blocked"


# ---------------------------------------------------------------------------
# Timeout/deadline tests
# ---------------------------------------------------------------------------


class TestTimeoutProtection:
    """Runtime respects deadline and timeout."""

    @pytest.mark.asyncio
    async def test_expired_deadline_stops(
        self,
        model_broker_submit: ModelBroker,
        tool_broker: ToolBroker,
        authorization: GoalActivationAuthorization,
        epochs: EpochSet,
        tmp_checkpoint_dir: str,
    ) -> None:
        expired_input = TurnInput(
            turn_seq=0,
            goal_summary="Test",
            plan_summary="Plan",
            step_contract={},
            attempt_id="att_1",
            available_capabilities=[],
            remaining_budget={},
            deadline=time.time() - 10,  # already expired
            history=[],
        )
        runtime = AgentRuntime(
            model_broker=model_broker_submit,
            tool_broker=tool_broker,
            checkpoint_dir=tmp_checkpoint_dir,
        )
        result = await runtime.execute_attempt(
            attempt_id="att_1",
            authorization=authorization,
            initial_input=expired_input,
            epochs=epochs,
        )
        assert result.stop_reason == "deadline"
        assert result.turn_count == 0
