"""Integration tests: Runtime with real ModelBroker and ToolBroker wired together."""

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
)
from src.autonomous.policy.budget_manager import BudgetManager
from src.autonomous.runtime.runtime import AgentRuntime, TurnInput

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def budget_manager() -> BudgetManager:
    mgr = BudgetManager()
    mgr.get_or_create_ledger(
        run_id="run_int",
        goal_id="goal_int",
        limits={"model_cost": 50.0, "tool_calls": 20.0},
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
        goal_id="goal_int",
        expires_at=time.time() + 3600,
    )


@pytest.fixture
def epochs() -> EpochSet:
    return EpochSet(plan_epoch=1)


# ---------------------------------------------------------------------------
# Integration: Model + Tool broker wired into Runtime
# ---------------------------------------------------------------------------


class TestRuntimeWithBrokers:
    """Full integration: model calls go through ModelBroker, tools through ToolBroker."""

    @pytest.mark.asyncio
    async def test_end_to_end_tool_execution(
        self, budget_manager: BudgetManager, ledger_id: str,
        authorization: GoalActivationAuthorization,
        epochs: EpochSet, tmp_dir: str,
    ) -> None:
        """Model proposes tool -> ToolBroker dispatches -> result fed back -> model submits."""
        call_count = {"n": 0}

        async def model_fn(prompt: dict) -> dict:
            call_count["n"] += 1
            history = prompt.get("history", [])
            if len(history) == 0:
                return {
                    "output_type": "tool_proposal",
                    "content": {},
                    "tool_proposals": [
                        {"capability": "list_files", "arguments": {"dir": "/tmp"}}
                    ],
                }
            return {
                "output_type": "submit_output",
                "content": {"files": ["a.txt", "b.txt"]},
            }

        # Set up tool broker with real capability
        tool_adapter = AsyncMock(return_value={"files": ["a.txt", "b.txt"]})
        registry = CapabilityRegistry()
        registry.register(
            CapabilityDescriptor(
                capability_id="list_files",
                name="List Files",
                risk_level=RiskLevel.R0,
            ),
            adapter=tool_adapter,
        )
        tool_broker = ToolBroker(
            registry=registry,
            policy_check_fn=lambda req, desc: {"decision": "allow"},
            budget_reserve_fn=lambda run_id, dim, amt: MagicMock(entry_id="bud_t1"),
            budget_settle_fn=lambda entry, amt: True,
            kill_check_fn=lambda cap: True,
            epoch_check_fn=lambda run_id, epoch, epochs: True,
        )

        model_broker = ModelBroker(
            model_fn=model_fn,
            budget_manager=budget_manager,
            ledger_id=ledger_id,
        )

        runtime = AgentRuntime(
            model_broker=model_broker,
            tool_broker=tool_broker,
            checkpoint_dir=tmp_dir,
        )

        initial_input = TurnInput(
            turn_seq=0,
            goal_summary="List files in /tmp",
            plan_summary="Step 1: list, Step 2: report",
            step_contract={"step_id": "s1"},
            attempt_id="att_int_1",
            available_capabilities=["list_files"],
            remaining_budget={"model_cost": 50.0},
            deadline=time.time() + 300,
            history=[],
        )

        result = await runtime.execute_attempt(
            attempt_id="att_int_1",
            authorization=authorization,
            initial_input=initial_input,
            epochs=epochs,
            run_id="run_int",
            step_id="s1",
        )

        assert result.stop_reason == "completed"
        assert result.turn_count == 2
        assert call_count["n"] == 2
        # Tool adapter was actually called
        tool_adapter.assert_called_once_with({"dir": "/tmp"})

    @pytest.mark.asyncio
    async def test_budget_shared_between_model_and_tools(
        self, authorization: GoalActivationAuthorization, epochs: EpochSet, tmp_dir: str,
    ) -> None:
        """Model broker and tool broker both draw from the same budget awareness."""
        # Tight budget: only 2 model calls allowed
        tight_mgr = BudgetManager()
        tight_ledger = tight_mgr.get_or_create_ledger(
            run_id="run_tight",
            goal_id="goal_tight",
            limits={"model_cost": 2.0},
        )
        tight_lid = tight_ledger.ledger_id

        call_count = {"n": 0}

        async def model_fn(prompt: dict) -> dict:
            call_count["n"] += 1
            return {
                "output_type": "tool_proposal",
                "content": {},
                "tool_proposals": [{"capability": "noop", "arguments": {}}],
            }

        registry = CapabilityRegistry()
        registry.register(
            CapabilityDescriptor(capability_id="noop", name="Noop", risk_level=RiskLevel.R0),
            adapter=AsyncMock(return_value={}),
        )
        tool_broker = ToolBroker(
            registry=registry,
            policy_check_fn=lambda req, desc: {"decision": "allow"},
            budget_reserve_fn=lambda run_id, dim, amt: MagicMock(entry_id="bud_n"),
            budget_settle_fn=lambda entry, amt: True,
            kill_check_fn=lambda cap: True,
            epoch_check_fn=lambda run_id, epoch, epochs: True,
        )

        model_broker = ModelBroker(
            model_fn=model_fn,
            budget_manager=tight_mgr,
            ledger_id=tight_lid,
            cost_per_call=1.0,
        )

        runtime = AgentRuntime(
            model_broker=model_broker,
            tool_broker=tool_broker,
            checkpoint_dir=tmp_dir,
            max_turns=10,
        )

        initial_input = TurnInput(
            turn_seq=0,
            goal_summary="Test budget",
            plan_summary="",
            step_contract={},
            attempt_id="att_budget",
            available_capabilities=["noop"],
            remaining_budget={},
            deadline=time.time() + 60,
            history=[],
        )

        result = await runtime.execute_attempt(
            attempt_id="att_budget",
            authorization=authorization,
            initial_input=initial_input,
            epochs=epochs,
            run_id="run_tight",
        )

        # Should stop after budget runs out (2 calls)
        assert result.stop_reason == "budget_exceeded"
        assert call_count["n"] == 2

    @pytest.mark.asyncio
    async def test_model_broker_ledger_tracks_calls(
        self, budget_manager: BudgetManager, ledger_id: str,
        authorization: GoalActivationAuthorization,
        epochs: EpochSet, tmp_dir: str,
    ) -> None:
        """Model broker ledger records all calls made during runtime execution."""

        async def model_fn(prompt: dict) -> dict:
            return {"output_type": "submit_output", "content": {"done": True}}

        model_broker = ModelBroker(
            model_fn=model_fn,
            budget_manager=budget_manager,
            ledger_id=ledger_id,
        )
        registry = CapabilityRegistry()
        tool_broker = ToolBroker(
            registry=registry,
            policy_check_fn=lambda req, desc: {"decision": "allow"},
            budget_reserve_fn=lambda run_id, dim, amt: MagicMock(entry_id="bud_x"),
            budget_settle_fn=lambda entry, amt: True,
            kill_check_fn=lambda cap: True,
            epoch_check_fn=lambda run_id, epoch, epochs: True,
        )

        runtime = AgentRuntime(
            model_broker=model_broker,
            tool_broker=tool_broker,
            checkpoint_dir=tmp_dir,
        )

        initial_input = TurnInput(
            turn_seq=0,
            goal_summary="Ledger test",
            plan_summary="",
            step_contract={},
            attempt_id="att_ledger",
            available_capabilities=[],
            remaining_budget={},
            deadline=time.time() + 60,
            history=[],
        )

        await runtime.execute_attempt(
            attempt_id="att_ledger",
            authorization=authorization,
            initial_input=initial_input,
            epochs=epochs,
            run_id="run_ledger",
        )

        # The model broker ledger should have 1 committed call
        ledger = model_broker.get_ledger("run_ledger")
        assert len(ledger) == 1
        assert ledger[0].state.value == "committed"
