"""Tests for ProjectContext.allowed_chat_ids true LRU semantics."""

from collections import OrderedDict
from unittest.mock import MagicMock, patch

import pytest

from src.project.context import ProjectContext


@pytest.fixture(autouse=True)
def _mock_settings():
    """Provide a Settings mock with max_allowed_chat_ids=3 for small tests."""
    settings = MagicMock()
    settings.max_allowed_chat_ids = 3
    settings.max_evicted_cache = 200
    with patch("src.config.get_settings", return_value=settings):
        yield settings


@pytest.fixture
def ctx(tmp_path):
    return ProjectContext(
        project_id="test_proj",
        project_name="TestProject",
        root_path=str(tmp_path),
        owner_chat_id="owner_chat",
        allowed_chat_ids=OrderedDict([("owner_chat", 1000.0)]),
    )


class TestLRUEvictionOrder:
    """AC-01: oldest non-owner chat_id is evicted first."""

    def test_eviction_order(self, ctx):
        ctx.add_chat_id("chat_A")
        ctx.add_chat_id("chat_B")
        # At capacity (3): owner_chat, chat_A, chat_B
        # Adding chat_C should evict chat_A (oldest non-owner)
        evicted = ctx.add_chat_id("chat_C")
        assert evicted == "chat_A"
        assert "chat_A" not in ctx.allowed_chat_ids
        assert "owner_chat" in ctx.allowed_chat_ids
        assert "chat_B" in ctx.allowed_chat_ids
        assert "chat_C" in ctx.allowed_chat_ids

    def test_owner_never_evicted(self, ctx):
        ctx.add_chat_id("chat_A")
        ctx.add_chat_id("chat_B")
        evicted = ctx.add_chat_id("chat_C")
        assert evicted == "chat_A"
        assert "owner_chat" in ctx.allowed_chat_ids

    def test_no_eviction_under_limit(self, ctx):
        evicted = ctx.add_chat_id("chat_A")
        assert evicted is None
        assert len(ctx.allowed_chat_ids) == 2


class TestLRUMoveToEnd:
    """AC-01: re-accessing moves to end, so older untouched entries are evicted."""

    def test_move_to_end_prevents_eviction(self, ctx):
        ctx.add_chat_id("chat_A")
        ctx.add_chat_id("chat_B")
        # At capacity: [owner_chat, chat_A, chat_B]
        # Re-access chat_A → move to end
        ctx.add_chat_id("chat_A")
        # Now order: [owner_chat, chat_B, chat_A]
        # Adding chat_C should evict chat_B (now oldest non-owner)
        evicted = ctx.add_chat_id("chat_C")
        assert evicted == "chat_B"
        assert "chat_A" in ctx.allowed_chat_ids
        assert "chat_B" not in ctx.allowed_chat_ids

    def test_move_to_end_returns_none(self, ctx):
        ctx.add_chat_id("chat_A")
        result = ctx.add_chat_id("chat_A")  # Already present
        assert result is None


class TestSnapshotPreservesOrder:
    """AC-02: to_snapshot → from_snapshot preserves LRU order."""

    def test_roundtrip_order(self, ctx):
        ctx.add_chat_id("chat_A")
        ctx.add_chat_id("chat_B")

        snap = ctx.to_snapshot()
        restored = ProjectContext.from_snapshot(snap)

        assert len(restored.allowed_chat_ids) == len(ctx.allowed_chat_ids)
        for (orig_cid, orig_ts), (rest_cid, rest_ts) in zip(
            ctx.allowed_chat_ids.items(), restored.allowed_chat_ids.items()
        ):
            assert orig_cid == rest_cid  # same chat_id
            assert abs(orig_ts - rest_ts) < 0.001  # same timestamp

    def test_snapshot_format(self, ctx):
        ctx.add_chat_id("chat_A")
        snap = ctx.to_snapshot()
        # Should be list of [chat_id, timestamp] pairs
        for entry in snap["allowed_chat_ids"]:
            assert isinstance(entry, list)
            assert len(entry) == 2
            assert isinstance(entry[0], str)
            assert isinstance(entry[1], float)


class TestSnapshotBackwardCompat:
    """AC-02: from_snapshot loads legacy list[str] format."""

    def test_legacy_string_list(self, tmp_path):
        data = {
            "project_id": "p1",
            "project_name": "P1",
            "root_path": str(tmp_path),
            "owner_chat_id": "owner",
            "allowed_chat_ids": ["owner", "chat_A", "chat_B"],
        }
        ctx = ProjectContext.from_snapshot(data)
        ids = list(ctx.allowed_chat_ids.keys())
        assert ids == ["owner", "chat_A", "chat_B"]
        # Timestamps should be monotonically increasing
        timestamps = list(ctx.allowed_chat_ids.values())
        assert timestamps == sorted(timestamps)
        assert timestamps[0] < timestamps[1] < timestamps[2]

    def test_empty_legacy(self, tmp_path):
        data = {
            "project_id": "p1",
            "project_name": "P1",
            "root_path": str(tmp_path),
            "allowed_chat_ids": [],
        }
        ctx = ProjectContext.from_snapshot(data)
        assert ctx.allowed_chat_ids == OrderedDict()

    def test_missing_field(self, tmp_path):
        data = {
            "project_id": "p1",
            "project_name": "P1",
            "root_path": str(tmp_path),
        }
        ctx = ProjectContext.from_snapshot(data)
        assert ctx.allowed_chat_ids == OrderedDict()


class TestChatIdSet:
    """Helper method _chat_id_set."""

    def test_returns_frozenset_of_ids(self, ctx):
        ctx.add_chat_id("chat_A")
        s = ctx._chat_id_set()
        assert isinstance(s, frozenset)
        assert "owner_chat" in s
        assert "chat_A" in s

    def test_ordered_dict_supports_o1_membership(self, ctx):
        """OrderedDict supports O(1) ``in`` checks directly."""
        assert isinstance(ctx.allowed_chat_ids, OrderedDict)
        ctx.add_chat_id("chat_A")
        assert isinstance(ctx.allowed_chat_ids, OrderedDict)
        assert "chat_A" in ctx.allowed_chat_ids

    def test_chat_id_set_reflects_mutations(self, ctx):
        """_chat_id_set() reflects latest state after add_chat_id."""
        ctx.add_chat_id("chat_A")
        old_snapshot = ctx._chat_id_set()
        ctx.add_chat_id("chat_B")
        new_snapshot = ctx._chat_id_set()
        # Old snapshot is a frozenset snapshot — unchanged
        assert "chat_B" not in old_snapshot
        # New snapshot reflects mutation
        assert "chat_B" in new_snapshot


class TestAllOwnerEvictionBoundary:
    """AC-R05: when all entries are owner, add_chat_id rejects over-limit."""

    def test_all_owner_rejects_over_limit(self, _mock_settings, tmp_path):
        """When limit=1 and the only entry is owner, new chat_id is rejected."""
        _mock_settings.max_allowed_chat_ids = 1
        ctx = ProjectContext(
            project_id="p",
            project_name="P",
            root_path=str(tmp_path),
            owner_chat_id="owner",
            allowed_chat_ids=OrderedDict([("owner", 1000.0)]),
        )
        result = ctx.add_chat_id("newcomer")
        from src.project.context import ADD_CHAT_ID_REJECTED
        assert result == ADD_CHAT_ID_REJECTED
        assert len(ctx.allowed_chat_ids) <= 1
        assert "newcomer" not in ctx.allowed_chat_ids
        assert "owner" in ctx.allowed_chat_ids

    def test_all_owner_does_not_exceed_limit(self, _mock_settings, tmp_path):
        """Invariant: dict never exceeds max_allowed_chat_ids."""
        _mock_settings.max_allowed_chat_ids = 2
        ctx = ProjectContext(
            project_id="p",
            project_name="P",
            root_path=str(tmp_path),
            owner_chat_id="owner",
            allowed_chat_ids=OrderedDict([("owner", 1000.0)]),
        )
        result = ctx.add_chat_id("newcomer")
        assert result is None
        assert len(ctx.allowed_chat_ids) <= 2


class TestEvictedChatIdsTracking:
    """AC-R19: evicted chat_ids are tracked in evicted_chat_ids set."""

    def test_evicted_chat_ids_tracked(self, ctx):
        ctx.add_chat_id("chat_A")
        ctx.add_chat_id("chat_B")
        # At capacity, adding chat_C evicts chat_A
        ctx.add_chat_id("chat_C")
        assert "chat_A" in ctx.evicted_chat_ids

    def test_evicted_chat_ids_not_serialized(self, ctx):
        ctx.add_chat_id("chat_A")
        ctx.add_chat_id("chat_B")
        ctx.add_chat_id("chat_C")  # evicts chat_A
        snap = ctx.to_snapshot()
        assert "evicted_chat_ids" not in snap

    def test_evicted_chat_ids_empty_after_from_snapshot(self, ctx):
        ctx.add_chat_id("chat_A")
        ctx.add_chat_id("chat_B")
        ctx.add_chat_id("chat_C")  # evicts chat_A
        snap = ctx.to_snapshot()
        restored = ProjectContext.from_snapshot(snap)
        assert restored.evicted_chat_ids == OrderedDict()


class TestConcurrentAddChatId:
    """AC-R05: concurrent add_chat_id must not produce data races."""

    def test_concurrent_add_chat_id_no_duplicates(self, _mock_settings, tmp_path):
        """10 threads concurrently calling add_chat_id — no duplicates, no over-limit."""
        _mock_settings.max_allowed_chat_ids = 5
        ctx = ProjectContext(
            project_id="p",
            project_name="P",
            root_path=str(tmp_path),
            owner_chat_id="owner",
            allowed_chat_ids=OrderedDict([("owner", 1000.0)]),
        )
        import threading

        barrier = threading.Barrier(10)
        errors: list[str] = []

        def worker(i: int):
            try:
                barrier.wait(timeout=5)
                ctx.add_chat_id(f"chat_{i}")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"
        # Invariant: never exceeds limit
        assert len(ctx.allowed_chat_ids) <= 5
        # No duplicate chat_ids
        ids = list(ctx.allowed_chat_ids.keys())
        assert len(ids) == len(set(ids)), f"Duplicates found: {ids}"
