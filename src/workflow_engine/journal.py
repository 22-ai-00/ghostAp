"""WorkflowJournal — persistent cache for agent() call results."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from .constants import DEFAULT_CACHE_MAX_ENTRIES
from .models import AgentCallResult, JournalEntry
from .storage import workflow_journal_dir

logger = logging.getLogger(__name__)


class WorkflowJournal:
    """Persistent cache for agent() call results.

    Each call is keyed by sha256(prompt|tool|model). Re-running a modified
    workflow script only executes changed parts; unchanged calls return cached
    results instantly.

    Storage layout::

        ~/.ghostap/workflow/projects/{absolute-project-path}/journal/{run_id}/
            _index.json          # full-key -> filename mapping
            {key[:16]}.json      # individual JournalEntry serialized

    Thread safety: A threading.Lock guards all mutations and LRU order
    updates. Reads also take the lock briefly while refreshing LRU order.

    The in-memory cache is bounded by ``max_entries``. When the cap is
    exceeded the least-recently-used entry is evicted from memory (its disk
    entry is preserved, so a later miss will reload it from disk). Passing
    ``max_entries=0`` disables the bound (original unbounded behaviour).
    """

    def __init__(
        self,
        root_path: str,
        run_id: str,
        max_entries: int = DEFAULT_CACHE_MAX_ENTRIES,
    ) -> None:
        self._root_path = root_path
        self._run_id = run_id
        self._journal_dir = Path(workflow_journal_dir(root_path)) / run_id
        self._index_path = self._journal_dir / "_index.json"
        self._max_entries = max_entries

        # In-memory LRU cache for fast repeated lookups (OrderedDict preserves
        # insertion order; newest keys live at the end).
        self._cache: "OrderedDict[str, AgentCallResult]" = OrderedDict()
        # Disk index: full_key -> filename (without .json suffix stored with it)
        self._index: dict[str, str] = {}

        # Stats tracking
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

        # Thread safety for mutations and LRU reordering
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

        # Load existing index from disk if present
        self._load_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def compute_key(
        prompt: str,
        tool: str,
        model: str | None,
        role: str | None = None,
        output_schema: dict | None = None,
    ) -> str:
        """Compute cache key as sha256 hex of 'prompt|tool|model|role|schema'."""
        schema_part = json.dumps(output_schema, sort_keys=True) if output_schema else ""
        raw = f"{prompt}|{tool}|{model or ''}|{role or ''}|{schema_part}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get_cached(self, key: str) -> Optional[AgentCallResult]:
        """Look up a cached result by key.

        Checks in-memory cache first, then falls back to disk.
        Returns None on cache miss or corrupted entry.
        On a successful hit the key is promoted to the most-recently-used end.
        """
        with self._lock:
            # Fast path: in-memory
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            # Check index for disk entry
            filename = self._index.get(key)
            if filename is None:
                self._misses += 1
                return None

        # Disk read outside lock (I/O bound, don't hold lock for file ops)
        entry_path = self._journal_dir / filename
        if not entry_path.exists():
            with self._lock:
                self._misses += 1
            return None

        try:
            data = json.loads(entry_path.read_text(encoding="utf-8"))
            entry = JournalEntry.model_validate(data)
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            logger.warning(
                "Corrupted journal entry %s for key %s: %s",
                filename,
                key[:16],
                exc,
            )
            with self._lock:
                self._misses += 1
            return None

        # Populate in-memory cache as most-recently-used, possibly evicting
        # the oldest entry if the cap is reached.
        with self._lock:
            self._cache[key] = entry.result
            self._cache.move_to_end(key)
            self._enforce_cap_locked()
            self._hits += 1
        return entry.result

    def store(self, key: str, result: AgentCallResult) -> None:
        """Store a result in both memory and disk."""
        with self._lock:
            # Ensure directory exists
            self._journal_dir.mkdir(parents=True, exist_ok=True)

            # Update in-memory cache (LRU — move/newest at end)
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = result
            self._enforce_cap_locked()

            # Determine filename — use key prefix, resolve collisions via index
            filename = f"{key[:16]}.json"

            # Check for filename collision with a different key
            for existing_key, existing_filename in self._index.items():
                if existing_filename == filename and existing_key != key:
                    # Collision: use longer prefix for the new key
                    filename = f"{key[:32]}.json"
                    break

            # Write entry to disk
            entry = JournalEntry(key=key, result=result, timestamp=time.time())
            entry_path = self._journal_dir / filename
            try:
                entry_path.write_text(
                    json.dumps(entry.model_dump(mode="json"), ensure_ascii=False),
                    encoding="utf-8",
                )
            except OSError as exc:
                logger.error("Failed to write journal entry %s: %s", filename, repr(exc))
                return

            # Update index
            self._index[key] = filename
            self._flush_index()

    def has(self, key: str) -> bool:
        """Check if a key exists in the journal (memory or disk index)."""
        return key in self._cache or key in self._index

    def clear(self) -> None:
        """Delete the entire run journal directory."""
        with self._lock:
            self._cache.clear()
            self._index.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

            if self._journal_dir.exists():
                try:
                    shutil.rmtree(self._journal_dir)
                except OSError as exc:
                    logger.error(
                        "Failed to clear journal directory %s: %s",
                        self._journal_dir,
                        exc,
                    )

    def stats(self) -> dict:
        """Return journal statistics."""
        return {
            "total": len(self._index),
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enforce_cap_locked(self) -> None:
        """Evict least-recently-used entries until ``len(_cache) <= max_entries``.

        Must be called while holding ``self._lock``. Only memory cache entries
        are evicted; the corresponding disk index and entry files are kept so
        the entry can still be reloaded on a future miss.
        """
        if not self._max_entries:
            return
        while len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)
            self._evictions += 1

    def _load_index(self) -> None:
        """Load the index file from disk if it exists."""
        if not self._index_path.exists():
            return

        try:
            data = json.loads(self._index_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._index = data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Corrupted journal index at %s: %s", self._index_path, repr(exc))
            self._index = {}

    def _flush_index(self) -> None:
        """Persist the index mapping to disk. Must be called under self._lock."""
        try:
            self._index_path.write_text(
                json.dumps(self._index, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.error("Failed to write journal index: %s", repr(exc))
