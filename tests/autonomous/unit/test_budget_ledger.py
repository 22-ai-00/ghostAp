"""Tests for BudgetManager: NaN, negative, concurrent oversell, version-CAS."""


import pytest

from src.autonomous.policy.budget_manager import (
    BudgetCASError,
    BudgetEntryNotFoundError,
    BudgetError,
    BudgetManager,
    BudgetOverdraftError,
    BudgetValidationError,
)


@pytest.fixture
def manager() -> BudgetManager:
    return BudgetManager()


@pytest.fixture
def ledger_with_budget(manager: BudgetManager):
    """Create a ledger with 1000 units in 'tokens' dimension."""
    ledger = manager.get_or_create_ledger(
        tenant_key="t1",
        run_id="run_1",
        goal_id="goal_1",
        limits={"tokens": 1000.0, "api_calls": 50.0},
    )
    return ledger


class TestAmountValidation:
    """Budget rejects NaN, negative, and invalid amounts."""

    def test_nan_amount_rejected(self, manager, ledger_with_budget) -> None:
        with pytest.raises(BudgetValidationError, match="NaN"):
            manager.reserve(
                ledger_with_budget.ledger_id, "tokens", float("nan")
            )

    def test_negative_amount_rejected(
        self, manager, ledger_with_budget
    ) -> None:
        with pytest.raises(BudgetValidationError, match="negative"):
            manager.reserve(ledger_with_budget.ledger_id, "tokens", -10.0)

    def test_infinity_amount_rejected(
        self, manager, ledger_with_budget
    ) -> None:
        with pytest.raises(BudgetValidationError, match="finite"):
            manager.reserve(
                ledger_with_budget.ledger_id, "tokens", float("inf")
            )

    def test_negative_infinity_rejected(
        self, manager, ledger_with_budget
    ) -> None:
        with pytest.raises(BudgetValidationError, match="finite"):
            manager.reserve(
                ledger_with_budget.ledger_id, "tokens", float("-inf")
            )

    def test_zero_amount_rejected(self, manager, ledger_with_budget) -> None:
        with pytest.raises(BudgetValidationError, match="positive"):
            manager.reserve(ledger_with_budget.ledger_id, "tokens", 0.0)

    def test_boolean_amount_rejected(
        self, manager, ledger_with_budget
    ) -> None:
        with pytest.raises(BudgetValidationError, match="numeric"):
            manager.reserve(ledger_with_budget.ledger_id, "tokens", True)  # type: ignore

    def test_nan_in_limits_rejected(self, manager) -> None:
        with pytest.raises(BudgetValidationError, match="NaN"):
            manager.get_or_create_ledger(
                run_id="run_nan",
                goal_id="goal_nan",
                limits={"tokens": float("nan")},
            )

    def test_settle_nan_actual_rejected(
        self, manager, ledger_with_budget
    ) -> None:
        entry_id = manager.reserve(
            ledger_with_budget.ledger_id, "tokens", 100.0
        )
        with pytest.raises(BudgetValidationError, match="NaN"):
            manager.settle(entry_id, actual_amount=float("nan"))


class TestReserveSettleRelease:
    """Core reserve/settle/release lifecycle."""

    def test_reserve_returns_entry_id(
        self, manager, ledger_with_budget
    ) -> None:
        entry_id = manager.reserve(
            ledger_with_budget.ledger_id, "tokens", 100.0
        )
        assert entry_id.startswith("bud_")

    def test_reserve_reduces_available(
        self, manager, ledger_with_budget
    ) -> None:
        manager.reserve(ledger_with_budget.ledger_id, "tokens", 300.0)
        ledger = manager.get_ledger(ledger_with_budget.ledger_id)
        assert ledger is not None
        assert ledger.available("tokens") == 700.0

    def test_settle_completes_reservation(
        self, manager, ledger_with_budget
    ) -> None:
        entry_id = manager.reserve(
            ledger_with_budget.ledger_id, "tokens", 100.0
        )
        manager.settle(entry_id)
        ledger = manager.get_ledger(ledger_with_budget.ledger_id)
        # Settled entries still count against budget
        assert ledger.available("tokens") == 900.0

    def test_settle_with_actual_amount(
        self, manager, ledger_with_budget
    ) -> None:
        entry_id = manager.reserve(
            ledger_with_budget.ledger_id, "tokens", 200.0
        )
        manager.settle(entry_id, actual_amount=150.0)
        ledger = manager.get_ledger(ledger_with_budget.ledger_id)
        # Settled at 150, not 200
        assert ledger.available("tokens") == 850.0

    def test_release_returns_budget(
        self, manager, ledger_with_budget
    ) -> None:
        entry_id = manager.reserve(
            ledger_with_budget.ledger_id, "tokens", 400.0
        )
        manager.release(entry_id)
        ledger = manager.get_ledger(ledger_with_budget.ledger_id)
        # Released entries don't count
        assert ledger.available("tokens") == 1000.0

    def test_double_settle_fails(self, manager, ledger_with_budget) -> None:
        entry_id = manager.reserve(
            ledger_with_budget.ledger_id, "tokens", 100.0
        )
        manager.settle(entry_id)
        with pytest.raises(BudgetError, match="cannot settle"):
            manager.settle(entry_id)

    def test_release_settled_fails(self, manager, ledger_with_budget) -> None:
        entry_id = manager.reserve(
            ledger_with_budget.ledger_id, "tokens", 100.0
        )
        manager.settle(entry_id)
        with pytest.raises(BudgetError, match="cannot release"):
            manager.release(entry_id)

    def test_nonexistent_entry_raises(self, manager) -> None:
        with pytest.raises(BudgetEntryNotFoundError):
            manager.settle("fake_entry_id")

    def test_nonexistent_ledger_raises(self, manager) -> None:
        with pytest.raises(BudgetEntryNotFoundError):
            manager.reserve("fake_ledger_id", "tokens", 100.0)


class TestConcurrentOversell:
    """Version-CAS prevents concurrent oversell."""

    def test_overdraft_rejected(self, manager, ledger_with_budget) -> None:
        with pytest.raises(BudgetOverdraftError):
            manager.reserve(
                ledger_with_budget.ledger_id, "tokens", 1001.0
            )

    def test_sequential_reservations_exhaust_budget(
        self, manager, ledger_with_budget
    ) -> None:
        manager.reserve(ledger_with_budget.ledger_id, "tokens", 500.0)
        manager.reserve(ledger_with_budget.ledger_id, "tokens", 500.0)
        with pytest.raises(BudgetOverdraftError):
            manager.reserve(ledger_with_budget.ledger_id, "tokens", 1.0)

    def test_version_cas_detects_stale_write(
        self, manager, ledger_with_budget
    ) -> None:
        ledger_id = ledger_with_budget.ledger_id
        # Read version at time 0
        initial_version = ledger_with_budget.aggregate_version

        # Another writer reserves (bumps version)
        manager.reserve(ledger_id, "tokens", 100.0)

        # Stale writer tries with old version
        with pytest.raises(BudgetCASError, match="version mismatch"):
            manager.reserve(
                ledger_id,
                "tokens",
                100.0,
                expected_version=initial_version,
            )

    def test_version_cas_succeeds_with_current(
        self, manager, ledger_with_budget
    ) -> None:
        ledger_id = ledger_with_budget.ledger_id
        manager.reserve(ledger_id, "tokens", 100.0)

        # Get current version
        current = manager.get_ledger(ledger_id)
        entry_id = manager.reserve(
            ledger_id,
            "tokens",
            50.0,
            expected_version=current.aggregate_version,
        )
        assert entry_id.startswith("bud_")

    def test_version_increments_on_each_mutation(
        self, manager, ledger_with_budget
    ) -> None:
        ledger_id = ledger_with_budget.ledger_id
        assert manager.get_ledger(ledger_id).aggregate_version == 0

        entry_id = manager.reserve(ledger_id, "tokens", 100.0)
        assert manager.get_ledger(ledger_id).aggregate_version == 1

        manager.settle(entry_id)
        assert manager.get_ledger(ledger_id).aggregate_version == 2


class TestUnknownBilling:
    """Conservative settlement of unknown billing."""

    def test_settle_unknown_billing(
        self, manager, ledger_with_budget
    ) -> None:
        entry_id = manager.settle_unknown_billing(
            ledger_with_budget.ledger_id, "tokens", 75.0
        )
        assert entry_id.startswith("bud_")
        ledger = manager.get_ledger(ledger_with_budget.ledger_id)
        assert ledger.available("tokens") == 925.0

    def test_unknown_billing_nan_rejected(
        self, manager, ledger_with_budget
    ) -> None:
        with pytest.raises(BudgetValidationError, match="NaN"):
            manager.settle_unknown_billing(
                ledger_with_budget.ledger_id, "tokens", float("nan")
            )


class TestEntryRestore:
    """Restore entries by ID."""

    def test_restore_released_entry(
        self, manager, ledger_with_budget
    ) -> None:
        entry_id = manager.reserve(
            ledger_with_budget.ledger_id, "tokens", 200.0
        )
        manager.release(entry_id)
        ledger = manager.get_ledger(ledger_with_budget.ledger_id)
        assert ledger.available("tokens") == 1000.0

        manager.restore_entry(entry_id)
        ledger = manager.get_ledger(ledger_with_budget.ledger_id)
        # Restored entry counts again
        assert ledger.available("tokens") == 800.0

    def test_restore_settled_entry(
        self, manager, ledger_with_budget
    ) -> None:
        entry_id = manager.reserve(
            ledger_with_budget.ledger_id, "tokens", 150.0
        )
        manager.settle(entry_id)
        manager.restore_entry(entry_id)
        ledger = manager.get_ledger(ledger_with_budget.ledger_id)
        # Entry is now reserved again
        assert ledger.available("tokens") == 850.0

    def test_restore_already_reserved_is_noop(
        self, manager, ledger_with_budget
    ) -> None:
        entry_id = manager.reserve(
            ledger_with_budget.ledger_id, "tokens", 100.0
        )
        manager.restore_entry(entry_id)  # Should not raise
        ledger = manager.get_ledger(ledger_with_budget.ledger_id)
        assert ledger.available("tokens") == 900.0

    def test_restore_nonexistent_raises(self, manager) -> None:
        with pytest.raises(BudgetEntryNotFoundError):
            manager.restore_entry("fake_id")


class TestLedgerLookup:
    """Ledger creation and retrieval."""

    def test_get_or_create_idempotent(self, manager) -> None:
        l1 = manager.get_or_create_ledger(
            run_id="r1", goal_id="g1", limits={"x": 100.0}
        )
        l2 = manager.get_or_create_ledger(
            run_id="r1", goal_id="g1", limits={"x": 200.0}
        )
        assert l1.ledger_id == l2.ledger_id

    def test_get_nonexistent_returns_none(self, manager) -> None:
        assert manager.get_ledger("nonexistent") is None
