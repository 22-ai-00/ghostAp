"""Sequence management for optimistic concurrency control."""

from __future__ import annotations

import threading


class SequenceManager:
    """Manages card_id → sequence counter with floor support.

    Each card has a monotonically increasing sequence number.
    When a sequence conflict (飞书 300317) is received, the floor is raised
    to avoid further conflicts.
    """

    def __init__(self) -> None:
        self._sequences: dict[str, int] = {}
        self._floors: dict[str, int] = {}
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

    def next_sequence(self, card_id: str) -> int:
        """Get the next sequence number for a card (atomic increment)."""
        with self._lock:
            current = self._sequences.get(card_id, 0)
            floor = self._floors.get(card_id, 0)
            next_val = max(current + 1, floor + 1)
            self._sequences[card_id] = next_val
            return next_val

    def raise_floor(self, card_id: str, floor: int) -> None:
        """Raise the sequence floor after a conflict.

        Next sequence will be at least floor + 1.
        """
        with self._lock:
            current_floor = self._floors.get(card_id, 0)
            if floor > current_floor:
                self._floors[card_id] = floor

    def reset(self, card_id: str) -> None:
        """Reset sequence for a card."""
        with self._lock:
            self._sequences.pop(card_id, None)
            self._floors.pop(card_id, None)
