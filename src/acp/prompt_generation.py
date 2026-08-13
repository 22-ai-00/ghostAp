"""Atomic prompt-generation provenance shared by sync transports."""

from __future__ import annotations

import threading


class PromptGenerationTracker:
    """Bind an explicit user stop to exactly one active prompt generation.

    The lazy initialization keeps older factories and lightweight test fixtures
    compatible while still making marker consumption and active-generation
    cleanup one atomic operation.
    """

    def _ensure_prompt_generation_state(self) -> threading.Lock:
        state = vars(self)
        lock = state.get("_prompt_generation_lock")
        if lock is None:
            lock = state.setdefault(
                "_prompt_generation_lock",
                threading.Lock(),  # leaf lock: never held while acquiring a LockLevel lock
            )
        with lock:
            state.setdefault("_prompt_generation", 0)
            state.setdefault("_active_prompt_generation", None)
            state.setdefault("_user_cancel_generation", None)
        return lock

    def _begin_prompt_generation(self) -> int:
        lock = self._ensure_prompt_generation_state()
        with lock:
            self._prompt_generation += 1
            generation = self._prompt_generation
            self._active_prompt_generation = generation
            return generation

    def _consume_prompt_generation(self, generation: int) -> bool:
        """Atomically consume cancellation provenance and retire a generation."""
        lock = self._ensure_prompt_generation_state()
        with lock:
            user_cancelled = self._user_cancel_generation == generation
            if self._active_prompt_generation == generation:
                self._active_prompt_generation = None
            if self._user_cancel_generation == generation:
                self._user_cancel_generation = None
            return user_cancelled

    def active_prompt_generation(self) -> int | None:
        """Return the exact prompt generation currently owned by this transport."""
        lock = self._ensure_prompt_generation_state()
        with lock:
            return self._active_prompt_generation

    def mark_user_cancel(self, generation: int) -> None:
        """Bind an explicit user cancellation to one active prompt generation."""
        lock = self._ensure_prompt_generation_state()
        with lock:
            if self._active_prompt_generation == generation:
                self._user_cancel_generation = generation


__all__ = ["PromptGenerationTracker"]
