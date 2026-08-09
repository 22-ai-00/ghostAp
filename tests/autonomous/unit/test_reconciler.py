"""Unit tests for Reconciler and Cleanup."""

from __future__ import annotations

import time

import pytest

from src.autonomous.domain import Effect, EffectState
from src.autonomous.supervisor.cleanup import (
    Cleanup,
    CleanupConfig,
    CleanupTarget,
    CleanupTargetType,
)
from src.autonomous.supervisor.reconciler import (
    Reconciler,
    ReconciliationActionType,
    ReconciliationReport,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeJournal:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def write_event(self, event_type: str, payload: dict) -> None:
        self.events.append((event_type, payload))


class FakeLease:
    def __init__(self, lease_id: str, step_id: str):
        self.lease_id = lease_id
        self.step_id = step_id


class FakeScheduler:
    def __init__(self, expired_leases: list[FakeLease] | None = None):
        self._expired = expired_leases or []
        self.released: list[tuple[str, bool]] = []

    async def check_expired_leases(self) -> list[FakeLease]:
        return self._expired

    async def release_lease(self, lease_id: str, success: bool) -> None:
        self.released.append((lease_id, success))


@pytest.fixture
def journal() -> FakeJournal:
    return FakeJournal()


@pytest.fixture
def scheduler() -> FakeScheduler:
    return FakeScheduler()


@pytest.fixture
def reconciler(journal: FakeJournal, scheduler: FakeScheduler) -> Reconciler:
    return Reconciler(
        journal=journal,
        scheduler=scheduler,
        effect_age_threshold=100.0,
        outbox_retry_threshold=60.0,
        trigger_stale_threshold=120.0,
    )


# ---------------------------------------------------------------------------
# Reconciler tests
# ---------------------------------------------------------------------------


class TestReconcileLeases:
    @pytest.mark.asyncio
    async def test_reclaim_expired_leases(self, journal: FakeJournal) -> None:
        expired = [FakeLease("lease_1", "step_1"), FakeLease("lease_2", "step_2")]
        scheduler = FakeScheduler(expired_leases=expired)
        reconciler = Reconciler(journal=journal, scheduler=scheduler)

        actions = await reconciler.reconcile_leases()
        assert len(actions) == 2
        assert actions[0].action_type == ReconciliationActionType.RECLAIM_LEASE.value
        assert actions[0].entity_id == "lease_1"
        assert scheduler.released == [("lease_1", False), ("lease_2", False)]

        event_types = [e[0] for e in journal.events]
        assert "reconciler.leases_reclaimed" in event_types

    @pytest.mark.asyncio
    async def test_no_expired_leases(self, reconciler: Reconciler) -> None:
        actions = await reconciler.reconcile_leases()
        assert len(actions) == 0


class TestReconcileEffects:
    @pytest.mark.asyncio
    async def test_mark_old_unknown_effects(self, reconciler: Reconciler) -> None:
        effects = [
            Effect(effect_id="eff_1", state=EffectState.UNKNOWN_EFFECT, created_at=time.time() - 200),
            Effect(effect_id="eff_2", state=EffectState.UNKNOWN_EFFECT, created_at=time.time() - 50),
            Effect(effect_id="eff_3", state=EffectState.COMMITTED),
        ]

        actions = await reconciler.reconcile_effects(effects)
        assert len(actions) == 1
        assert actions[0].entity_id == "eff_1"
        assert actions[0].action_type == ReconciliationActionType.MARK_UNKNOWN.value
        # Effect objects are frozen; reconciler only records actions, does not mutate

    @pytest.mark.asyncio
    async def test_no_unknown_effects(self, reconciler: Reconciler) -> None:
        effects = [Effect(effect_id="eff_1", state=EffectState.COMMITTED)]
        actions = await reconciler.reconcile_effects(effects)
        assert len(actions) == 0


class TestReconcileOutbox:
    @pytest.mark.asyncio
    async def test_retry_stale_outbox_entries(self, reconciler: Reconciler, journal: FakeJournal) -> None:
        entries = [
            {"entry_id": "out_1", "state": "pending", "created_at": time.time() - 120, "attempt_count": 1, "max_attempts": 5},
            {"entry_id": "out_2", "state": "failed", "created_at": time.time() - 120, "attempt_count": 5, "max_attempts": 5},
            {"entry_id": "out_3", "state": "delivered", "created_at": time.time() - 120},
        ]

        actions = await reconciler.reconcile_outbox(entries)
        assert len(actions) == 2

        action_map = {a.entity_id: a for a in actions}
        assert action_map["out_1"].action_type == ReconciliationActionType.RETRY_OUTBOX.value
        assert action_map["out_2"].action_type == ReconciliationActionType.DEAD_LETTER.value

        event_types = [e[0] for e in journal.events]
        assert "reconciler.outbox_reconciled" in event_types

    @pytest.mark.asyncio
    async def test_fresh_entries_not_retried(self, reconciler: Reconciler) -> None:
        entries = [
            {"entry_id": "out_1", "state": "pending", "created_at": time.time(), "attempt_count": 0, "max_attempts": 5},
        ]
        actions = await reconciler.reconcile_outbox(entries)
        assert len(actions) == 0


class TestReconcileTriggers:
    @pytest.mark.asyncio
    async def test_advance_stale_cursors(self, reconciler: Reconciler, journal: FakeJournal) -> None:
        triggers = [
            {"subscription_id": "trig_1", "active": True, "cursor": {"last_success_at": time.time() - 300}},
            {"subscription_id": "trig_2", "active": True, "cursor": {"last_success_at": time.time() - 10}},
            {"subscription_id": "trig_3", "active": False, "cursor": {"last_success_at": time.time() - 300}},
        ]

        actions = await reconciler.reconcile_triggers(triggers)
        assert len(actions) == 1
        assert actions[0].entity_id == "trig_1"
        assert actions[0].action_type == ReconciliationActionType.ADVANCE_CURSOR.value

    @pytest.mark.asyncio
    async def test_no_stale_triggers(self, reconciler: Reconciler) -> None:
        triggers = [
            {"subscription_id": "trig_1", "active": True, "cursor": {"last_success_at": time.time()}},
        ]
        actions = await reconciler.reconcile_triggers(triggers)
        assert len(actions) == 0


class TestReconcileOrphans:
    @pytest.mark.asyncio
    async def test_find_orphan_attempts(self, reconciler: Reconciler) -> None:
        attempts = [
            {"attempt_id": "att_1", "state": "active", "worker_id": "wkr_dead"},
            {"attempt_id": "att_2", "state": "active", "worker_id": "wkr_alive"},
            {"attempt_id": "att_3", "state": "succeeded", "worker_id": "wkr_dead"},
        ]
        active_workers = {"wkr_alive"}

        actions = await reconciler.reconcile_orphan_attempts(attempts, active_workers)
        assert len(actions) == 1
        assert actions[0].entity_id == "att_1"
        assert actions[0].action_type == ReconciliationActionType.RETRY_STEP.value


class TestFullReconciliation:
    @pytest.mark.asyncio
    async def test_full_reconciliation_report(self, journal: FakeJournal) -> None:
        expired = [FakeLease("lease_1", "step_1")]
        scheduler = FakeScheduler(expired_leases=expired)
        reconciler = Reconciler(
            journal=journal,
            scheduler=scheduler,
            effect_age_threshold=100.0,
            outbox_retry_threshold=60.0,
            trigger_stale_threshold=120.0,
        )

        effects = [Effect(effect_id="eff_1", state=EffectState.UNKNOWN_EFFECT, created_at=time.time() - 200)]
        outbox = [{"entry_id": "out_1", "state": "pending", "created_at": time.time() - 120, "attempt_count": 1, "max_attempts": 5}]
        triggers = [{"subscription_id": "trig_1", "active": True, "cursor": {"last_success_at": time.time() - 300}}]
        attempts = [{"attempt_id": "att_1", "state": "active", "worker_id": "wkr_dead"}]

        report = await reconciler.full_reconciliation(
            effects=effects,
            outbox_entries=outbox,
            triggers=triggers,
            attempts=attempts,
            active_workers=set(),
        )

        assert isinstance(report, ReconciliationReport)
        assert report.lease_actions == 1
        assert report.effect_actions == 1
        assert report.outbox_actions == 1
        assert report.trigger_actions == 1
        assert report.orphan_actions == 1

        event_types = [e[0] for e in journal.events]
        assert "reconciler.full_start" in event_types
        assert "reconciler.full_complete" in event_types

    @pytest.mark.asyncio
    async def test_action_history(self, reconciler: Reconciler) -> None:
        effects = [Effect(effect_id="eff_1", state=EffectState.UNKNOWN_EFFECT, created_at=time.time() - 200)]
        await reconciler.reconcile_effects(effects)
        assert reconciler.get_action_count() == 1
        assert len(reconciler.get_recent_actions()) == 1


# ---------------------------------------------------------------------------
# Cleanup tests
# ---------------------------------------------------------------------------


class TestCleanup:
    @pytest.mark.asyncio
    async def test_identify_completed_runs(self, journal: FakeJournal) -> None:
        config = CleanupConfig(completed_run_retention_seconds=100.0)
        cleanup = Cleanup(journal=journal, config=config)

        completed_runs = [
            {"run_id": "run_old", "state": "succeeded", "completed_at": time.time() - 200},
            {"run_id": "run_new", "state": "succeeded", "completed_at": time.time() - 10},
            {"run_id": "run_active", "state": "executing", "completed_at": time.time() - 200},
        ]

        targets = await cleanup.identify_cleanup_targets(completed_runs, [])
        assert len(targets) == 1
        assert targets[0].entity_id == "run_old"
        assert targets[0].target_type == CleanupTargetType.COMPLETED_RUN
        assert targets[0].safe_to_delete is True

    @pytest.mark.asyncio
    async def test_identify_dead_letters(self, journal: FakeJournal) -> None:
        config = CleanupConfig(dead_letter_retention_seconds=100.0)
        cleanup = Cleanup(journal=journal, config=config)

        dead_letters = [
            {"step_id": "step_old", "moved_at": time.time() - 200},
            {"step_id": "step_new", "moved_at": time.time() - 10},
        ]

        targets = await cleanup.identify_cleanup_targets([], dead_letters)
        assert len(targets) == 1
        assert targets[0].entity_id == "step_old"
        assert targets[0].target_type == CleanupTargetType.DEAD_LETTER

    @pytest.mark.asyncio
    async def test_execute_cleanup(self, journal: FakeJournal) -> None:
        cleanup = Cleanup(journal=journal)
        targets = [
            CleanupTarget(entity_id="run_1", target_type=CleanupTargetType.COMPLETED_RUN, safe_to_delete=True),
            CleanupTarget(entity_id="run_2", target_type=CleanupTargetType.COMPLETED_RUN, safe_to_delete=False),
        ]

        result = await cleanup.execute_cleanup(targets)
        assert result.targets_scanned == 2
        assert result.targets_archived == 1
        assert result.targets_skipped == 1

        archived = cleanup.get_archived()
        assert len(archived) == 1
        assert archived[0].entity_id == "run_1"

    @pytest.mark.asyncio
    async def test_dry_run_skips_all(self, journal: FakeJournal) -> None:
        config = CleanupConfig(dry_run=True)
        cleanup = Cleanup(journal=journal, config=config)
        targets = [
            CleanupTarget(entity_id="run_1", target_type=CleanupTargetType.COMPLETED_RUN, safe_to_delete=True),
        ]

        result = await cleanup.execute_cleanup(targets)
        assert result.targets_archived == 0
        assert result.targets_skipped == 1
        assert len(cleanup.get_archived()) == 0

    @pytest.mark.asyncio
    async def test_batch_size_limit(self, journal: FakeJournal) -> None:
        config = CleanupConfig(
            completed_run_retention_seconds=0.0,
            max_cleanup_batch_size=2,
        )
        cleanup = Cleanup(journal=journal, config=config)

        runs = [
            {"run_id": f"run_{i}", "state": "succeeded", "completed_at": time.time() - 100}
            for i in range(10)
        ]

        targets = await cleanup.identify_cleanup_targets(runs, [])
        assert len(targets) == 2

    @pytest.mark.asyncio
    async def test_cleanup_journal_events(self, journal: FakeJournal) -> None:
        cleanup = Cleanup(journal=journal)
        targets = [
            CleanupTarget(entity_id="run_1", target_type=CleanupTargetType.COMPLETED_RUN, safe_to_delete=True),
        ]
        await cleanup.execute_cleanup(targets)

        event_types = [e[0] for e in journal.events]
        assert "cleanup.batch_start" in event_types
        assert "cleanup.target_archived" in event_types
        assert "cleanup.batch_complete" in event_types

    @pytest.mark.asyncio
    async def test_cleanup_no_journal(self) -> None:
        cleanup = Cleanup(journal=None)
        targets = [
            CleanupTarget(entity_id="run_1", target_type=CleanupTargetType.COMPLETED_RUN, safe_to_delete=True),
        ]
        result = await cleanup.execute_cleanup(targets)
        assert result.targets_archived == 1
