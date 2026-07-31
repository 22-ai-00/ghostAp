"""SlockEngineManager — manages SlockEngine instances per chat+project."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..engine_base import BaseEngineManager
from .activation import slock_activation_guard
from .engine import SlockEngine
from .memory_manager import default_slock_storage_base

logger = logging.getLogger(__name__)
_LARK_CHAT_ID_RE = re.compile(r"^oc_[A-Za-z0-9_]+$")


class SlockEngineResolutionError(RuntimeError):
    """Activated Slock identity is absent, ambiguous, or mismatched."""


class MarkerArchiveError(OSError):
    """Marker rename durability failed; ``restored`` controls revoke cancel."""

    def __init__(self, message: str, *, restored: bool) -> None:
        super().__init__(message)
        self.restored = restored


@dataclass(frozen=True, slots=True)
class ActivatedSlockBinding:
    engine_identity: str
    chat_id: str
    root_identity: str
    canonical_root: str
    channel_id: str
    engine: SlockEngine = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class RevokeMarkerDisposition:
    chat_id: str
    state: str
    archived_path: str = ""


class SlockEngineManager(BaseEngineManager["SlockEngine"]):
    """Manages SlockEngine instances per chat+project.

    Thread-safe: all dict mutations are protected by _lock.
    Uses secondary index (_chat_keys) for efficient per-chat lookups.
    """

    def __init__(self, storage_base_path: str = "") -> None:
        super().__init__()
        self._managed_chats: set[str] = set()
        self._retired_chats: set[str] = set()
        self._dissolving_chats: set[str] = set()
        self._reserved_team_names: set[str] = set()
        self._storage_base_path = storage_base_path or default_slock_storage_base()
        self._blocked_team_names = self._load_pending_cleanup_names()

    def register_managed_chat(self, chat_id: str) -> None:
        """Declare a chat_id as managed by the slock engine (event routing)."""
        with self._lock:
            self._managed_chats.add(chat_id)

    def unregister_managed_chat(self, chat_id: str) -> None:
        """Remove a chat_id from slock management."""
        with self._lock:
            self._managed_chats.discard(chat_id)

    def claim_dissolve(self, chat_id: str) -> bool:
        """Claim the one in-flight dissolve transaction allowed per chat."""
        with self._lock:
            if chat_id in self._dissolving_chats:
                return False
            self._dissolving_chats.add(chat_id)
            return True

    def release_dissolve(self, chat_id: str) -> None:
        with self._lock:
            self._dissolving_chats.discard(chat_id)

    def reserve_team_name(self, team_name: str) -> bool:
        """Atomically reject active or concurrently-created duplicate names."""
        normalized = (team_name or "").strip().casefold()
        if not normalized:
            return False
        with self._lock:
            if normalized in self._reserved_team_names or normalized in self._blocked_team_names:
                return False
            for engine in self._engines.values():
                channel = engine.channel
                if channel and (channel.team_name or channel.name or "").strip().casefold() == normalized:
                    return False
            self._reserved_team_names.add(normalized)
            return True

    def reserve_retained_team_name(self, team_name: str) -> Optional[str]:
        """Claim an exact durable retained-group retry despite its name block."""

        chat_id = self.retained_team_chat_id(team_name)
        if chat_id is None:
            return None
        normalized = (team_name or "").strip().casefold()
        with self._lock:
            if normalized in self._reserved_team_names:
                return None
            for engine in self._engines.values():
                channel = engine.channel
                if (
                    channel
                    and (channel.team_name or channel.name or "").strip().casefold()
                    == normalized
                ):
                    return None
            self._reserved_team_names.add(normalized)
            return chat_id

    def retained_team_chat_id(self, team_name: str) -> Optional[str]:
        """Return the exact durable retained chat without changing reservation."""

        record = self._pending_cleanup_record(team_name)
        if record is None or record.get("delete_state") != "untrusted_retained":
            return None
        chat_id = record.get("chat_id")
        return chat_id if isinstance(chat_id, str) and chat_id else None

    def release_team_name(self, team_name: str) -> None:
        normalized = (team_name or "").strip().casefold()
        with self._lock:
            self._reserved_team_names.discard(normalized)

    def block_team_name_for_cleanup(
        self,
        team_name: str,
        chat_id: str,
        delete_state: str,
    ) -> bool:
        """Persist a residual-group tombstone and block same-name creation.

        Returns whether the block reached durable storage. The in-memory block
        is installed first, so callers remain safe even if persistence fails.
        """
        normalized = (team_name or "").strip().casefold()
        if not normalized:
            return False
        if delete_state not in {
            "create_outcome_unknown",
            "registry_bind_uncertain",
        }:
            with self._lock:
                self._blocked_team_names.add(normalized)

        records_dir = os.path.join(self._storage_base_path, "pending_cleanup")
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        record_path = os.path.join(records_dir, f"{digest}.json")
        temp_path = f"{record_path}.tmp.{os.getpid()}.{time.time_ns()}"
        record = {
            "version": 1,
            "team_name": (team_name or "").strip(),
            "chat_id": chat_id,
            "delete_state": delete_state,
            "created_at_ns": time.time_ns(),
        }
        try:
            records_dir_existed = os.path.isdir(records_dir)
            os.makedirs(records_dir, mode=0o700, exist_ok=True)
            if not records_dir_existed:
                self._fsync_directory(self._storage_base_path)
            with open(temp_path, "w", encoding="utf-8") as handle:
                json.dump(record, handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, record_path)
            self._fsync_directory(records_dir)
            return True
        except OSError:
            logger.exception(
                "block_team_name_for_cleanup: cannot persist residual group chat=%s",
                chat_id,
            )
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            return False

    def consume_retained_team_name(self, team_name: str, chat_id: str) -> bool:
        """Durably consume one exact retained-group block after activation."""

        record = self._pending_cleanup_record(team_name)
        if (
            record is None
            or record.get("delete_state") != "untrusted_retained"
            or record.get("chat_id") != chat_id
        ):
            return False
        normalized = (team_name or "").strip().casefold()
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        records_dir = os.path.join(self._storage_base_path, "pending_cleanup")
        record_path = os.path.join(records_dir, f"{digest}.json")
        try:
            os.unlink(record_path)
            self._fsync_directory(records_dir)
        except OSError:
            logger.exception(
                "consume_retained_team_name: cannot persist cleanup chat=%s",
                chat_id,
            )
            return False
        with self._lock:
            self._blocked_team_names.discard(normalized)
        return True

    def _pending_cleanup_record(self, team_name: str) -> Optional[dict]:
        normalized = (team_name or "").strip().casefold()
        if not normalized:
            return None
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        records_dir = os.path.join(self._storage_base_path, "pending_cleanup")
        record_path = os.path.join(records_dir, f"{digest}.json")
        if not os.path.isfile(record_path) or os.path.islink(record_path):
            return None
        try:
            with open(record_path, "r", encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        if (
            not isinstance(record, dict)
            or str(record.get("team_name") or "").strip().casefold()
            != normalized
        ):
            return None
        return record

    def _load_pending_cleanup_names(self) -> set[str]:
        """Load durable residual-group name blocks, ignoring unsafe records."""
        records_dir = os.path.join(self._storage_base_path, "pending_cleanup")
        if not os.path.isdir(records_dir) or os.path.islink(records_dir):
            return set()
        blocked: set[str] = set()
        try:
            entries = os.listdir(records_dir)
        except OSError:
            logger.exception("cannot list Slock pending-cleanup records")
            return blocked
        for entry in entries:
            if not re.fullmatch(r"[0-9a-f]{64}\.json", entry):
                continue
            path = os.path.join(records_dir, entry)
            if not os.path.isfile(path) or os.path.islink(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    record = json.load(handle)
            except (OSError, json.JSONDecodeError):
                logger.warning("skipping invalid Slock cleanup record: %s", path)
                continue
            normalized = str(record.get("team_name") or "").strip().casefold()
            expected = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if record.get("delete_state") == "create_outcome_unknown":
                created_at_ns = record.get("created_at_ns")
                if (
                    isinstance(created_at_ns, int)
                    and time.time_ns() - created_at_ns < 10 * 60 * 60 * 1_000_000_000
                ):
                    continue
            if record.get("delete_state") == "registry_bind_uncertain":
                continue
            if normalized and entry == f"{expected}.json":
                blocked.add(normalized)
        return blocked

    def archive_managed_chat_marker(self, chat_id: str) -> Optional[str]:
        """Atomically retire an active marker so restart cannot revive it."""
        if not isinstance(chat_id, str) or not _LARK_CHAT_ID_RE.fullmatch(chat_id):
            raise ValueError("invalid chat_id")
        groups_dir = os.path.realpath(os.path.join(self._storage_base_path, "groups"))
        group_dir = os.path.join(groups_dir, chat_id)
        if os.path.islink(group_dir) or os.path.commonpath((groups_dir, os.path.realpath(group_dir))) != groups_dir:
            raise ValueError("unsafe chat marker path")
        marker_path = os.path.join(group_dir, ".slock_channel.json")
        if not os.path.isfile(marker_path):
            return None
        archived_path = os.path.join(
            group_dir,
            f".slock_channel.dissolved.{time.time_ns()}.json",
        )
        os.replace(marker_path, archived_path)
        try:
            self._fsync_directory(group_dir)
        except OSError as archive_error:
            restored = False
            try:
                os.replace(archived_path, marker_path)
                self._fsync_directory(group_dir)
                restored = True
            except OSError:
                logger.exception(
                    "failed to restore marker after archive durability error chat=%s",
                    chat_id,
                )
            raise MarkerArchiveError(
                str(archive_error),
                restored=restored,
            ) from archive_error
        return archived_path

    def prepare_revoke_marker(self, chat_id: str) -> RevokeMarkerDisposition:
        """Idempotently ensure an active marker cannot be restored."""

        archived = self.archive_managed_chat_marker(chat_id)
        if archived is not None:
            return RevokeMarkerDisposition(chat_id, "archived", archived)
        group_dir = os.path.realpath(
            os.path.join(self._storage_base_path, "groups", chat_id)
        )
        try:
            candidates = sorted(
                os.path.join(group_dir, entry)
                for entry in os.listdir(group_dir)
                if re.fullmatch(r"\.slock_channel\.dissolved\.\d+\.json", entry)
                and os.path.isfile(os.path.join(group_dir, entry))
                and not os.path.islink(os.path.join(group_dir, entry))
            )
        except FileNotFoundError:
            candidates = []
        if candidates:
            return RevokeMarkerDisposition(chat_id, "archived", candidates[-1])
        return RevokeMarkerDisposition(chat_id, "absent")

    def restore_archived_chat_marker(self, chat_id: str, archived_path: str) -> None:
        """Restore a retired marker as compensation for a failed teardown."""
        if not isinstance(chat_id, str) or not _LARK_CHAT_ID_RE.fullmatch(chat_id):
            raise ValueError("invalid chat_id")
        group_dir = os.path.realpath(os.path.join(self._storage_base_path, "groups", chat_id))
        if os.path.dirname(os.path.realpath(archived_path)) != group_dir:
            raise ValueError("unsafe archived marker path")
        if not os.path.isfile(archived_path) or os.path.islink(archived_path):
            raise FileNotFoundError(archived_path)
        os.replace(archived_path, os.path.join(group_dir, ".slock_channel.json"))
        self._fsync_directory(group_dir)

    def restore_revoke_marker(
        self,
        disposition: RevokeMarkerDisposition,
    ) -> bool:
        if disposition.state == "absent":
            return True
        try:
            self.restore_archived_chat_marker(
                disposition.chat_id,
                disposition.archived_path,
            )
            return True
        except (OSError, ValueError):
            logger.exception(
                "failed to durably restore revoke marker chat=%s",
                disposition.chat_id,
            )
            return False

    @staticmethod
    def _fsync_directory(path: str) -> None:
        directory_fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def retire_deleted_chat(self, chat_id: str) -> Optional[str]:
        """Durably retire a remotely deleted chat and all indexed runtimes.

        Inactive engines remain indexed after a partial ``deactivate()``
        failure. Enumerating the full chat index lets a redelivered Lark event
        finish that interrupted teardown.
        """

        with slock_activation_guard():
            self._retired_chats.add(chat_id)
            archived_path = self.archive_managed_chat_marker(chat_id)
            for engine in tuple(self._iter_chat_engines(chat_id)):
                root_path = engine.root_path
                engine.deactivate()
                self.remove(chat_id, root_path)
            self.unregister_managed_chat(chat_id)
            return archived_path

    def is_managed_chat(self, chat_id: str) -> bool:
        """Check if a chat_id is managed by the slock engine."""
        with self._lock:
            return chat_id in self._managed_chats

    def _create_engine(
        self,
        chat_id: str,
        root_path: str,
        agent_type: str,
        engine_name: str,
        model_name: Optional[str],
    ) -> SlockEngine:
        return SlockEngine(
            chat_id=chat_id,
            root_path=root_path,
            agent_type=agent_type,
            engine_name=engine_name,
            model_name=model_name,
            memory_base_path=self._storage_base_path,
        )

    def get_or_create(
        self,
        chat_id: str,
        root_path: str,
        engine_name: str = "Coco",
        *,
        model_name: Optional[str] = None,
    ) -> SlockEngine:
        """Serialize manager membership changes with employee dispatch commits."""

        with slock_activation_guard():
            return super().get_or_create(
                chat_id,
                root_path,
                engine_name,
                model_name=model_name,
            )

    def get_or_create_activated(
        self,
        chat_id: str,
        root_path: str,
        channel,
        engine_name: str = "Coco",
        *,
        model_name: Optional[str] = None,
    ) -> SlockEngine:
        """Atomically create, activate, and register a live chat binding."""

        if channel.channel_id != chat_id:
            raise SlockEngineResolutionError("Slock channel binding mismatch")
        with slock_activation_guard():
            if chat_id in self._retired_chats:
                raise SlockEngineResolutionError("deleted Slock chat cannot be reactivated")
            with self._lock:
                committed = self._get_activated_engine_locked(chat_id)
            if committed is not None:
                return committed
            engine = super().get_or_create(
                chat_id,
                root_path,
                engine_name,
                model_name=model_name,
            )
            try:
                engine.activate_channel(channel)
                self.register_managed_chat(chat_id)
            except Exception:
                self._rollback_failed_activation(chat_id, root_path, engine)
                raise
            return engine

    def _rollback_failed_activation(
        self,
        chat_id: str,
        root_path: str,
        engine: SlockEngine,
    ) -> None:
        """Make a partially activated engine unreachable before retry."""

        try:
            engine.deactivate()
        except Exception:
            logger.exception("failed to deactivate partial Slock activation chat=%s", chat_id)
        try:
            engine.cleanup()
        except Exception:
            logger.exception("failed to clean partial Slock activation chat=%s", chat_id)
        key = f"{chat_id}:{root_path}"
        with self._lock:
            self._managed_chats.discard(chat_id)
            if self._engines.get(key) is engine:
                del self._engines[key]
                self._remove_index(chat_id, key)

    def remove(self, chat_id: str, root_path: str) -> None:
        """Remove and cleanup a slock engine instance."""
        key = f"{chat_id}:{root_path}"
        with slock_activation_guard(), self._lock:
            if key in self._engines:
                self._engines[key].cleanup()
                del self._engines[key]
                self._remove_index(chat_id, key)

    def discard_engine_for_recovery(self, chat_id: str, root_path: str) -> None:
        """Drop a possibly half-shutdown instance before compensation rebuild."""
        key = f"{chat_id}:{root_path}"
        with slock_activation_guard(), self._lock:
            self._engines.pop(key, None)
            self._remove_index(chat_id, key)

    def get_activated_engine(self, chat_id: str) -> Optional[SlockEngine]:
        """Get only a fully activated and event-routable chat binding."""

        with slock_activation_guard(), self._lock:
            return self._get_activated_engine_locked(chat_id)

    def _get_activated_engine_locked(self, chat_id: str) -> Optional[SlockEngine]:
        if chat_id not in self._managed_chats:
            return None
        for key in self._chat_keys.get(chat_id, ()):
            engine = self._engines.get(key)
            if engine is not None and engine.channel is not None:
                return engine
        return None

    @contextmanager
    def employee_activation_guard(
        self,
        *,
        chat_id: str,
        expected_root_identity: str | None = None,
    ):
        """Hold the global activation guard through caller's dispatch commit."""

        with slock_activation_guard(), self._lock:
            yield self._resolve_employee_engine_locked(
                chat_id=chat_id,
                expected_root_identity=expected_root_identity,
            )

    def resolve_employee_engine(
        self,
        *,
        chat_id: str,
        expected_root_identity: str | None = None,
    ) -> ActivatedSlockBinding:
        """Resolve exactly one already-activated chat and canonical root."""

        with self.employee_activation_guard(
            chat_id=chat_id,
            expected_root_identity=expected_root_identity,
        ) as binding:
            return binding

    def _resolve_employee_engine_locked(
        self,
        *,
        chat_id: str,
        expected_root_identity: str | None,
    ) -> ActivatedSlockBinding:
        if not _LARK_CHAT_ID_RE.fullmatch(chat_id):
            raise SlockEngineResolutionError("invalid employee Slock chat")
        if expected_root_identity is not None and not re.fullmatch(
            r"[0-9a-f]{64}", expected_root_identity
        ):
            raise SlockEngineResolutionError("invalid employee Slock root identity")
        engines = tuple(
            engine
            for key in self._chat_keys.get(chat_id, ())
            if (engine := self._engines.get(key)) is not None
            and engine.channel is not None
        )
        if chat_id not in self._managed_chats or len(engines) != 1:
            raise SlockEngineResolutionError(
                "employee Slock requires exactly one activated engine"
            )
        engine = engines[0]
        channel = engine.channel
        if channel is None or channel.channel_id != chat_id or engine.chat_id != chat_id:
            raise SlockEngineResolutionError("employee Slock channel binding mismatch")
        canonical_root = os.path.realpath(engine.root_path)
        root_identity = hashlib.sha256(canonical_root.encode("utf-8")).hexdigest()
        if expected_root_identity is not None and root_identity != expected_root_identity:
            raise SlockEngineResolutionError("employee Slock root binding mismatch")
        engine_identity = hashlib.sha256(
            f"{chat_id}\0{canonical_root}".encode("utf-8")
        ).hexdigest()
        return ActivatedSlockBinding(
            engine_identity=engine_identity,
            chat_id=chat_id,
            root_identity=root_identity,
            canonical_root=canonical_root,
            channel_id=channel.channel_id,
            engine=engine,
        )

    def is_slock_active(self, chat_id: str) -> bool:
        """Check if slock mode is active for a given chat."""
        return self.get_activated_engine(chat_id) is not None

    def list_activated_engines(self) -> list[SlockEngine]:
        """List all engines with an activated slock channel."""
        with slock_activation_guard(), self._lock:
            return [
                engine
                for chat_id in self._managed_chats
                for key in self._chat_keys.get(chat_id, ())
                if (engine := self._engines.get(key)) is not None
                and engine.channel is not None
            ]

    def find_team(self, name: str) -> Optional[SlockEngine]:
        """Find an activated team by team name, channel name, or chat id."""
        needle = (name or "").strip().lower()
        if not needle:
            return None
        for engine in self.list_activated_engines():
            channel = engine.channel
            if channel is None:
                continue
            candidates = {
                channel.channel_id.lower(),
                (channel.name or "").lower(),
                (channel.team_name or "").lower(),
            }
            if needle in candidates:
                return engine
        return None

    def restore_from_disk(
        self,
        root_path: str,
        *,
        managed_group_active: Callable[[str], bool] | None = None,
    ) -> int:
        """Scan app-level Slock group markers and restore engines.

        For each valid marker file, rebuilds a SlockEngine, activates the
        channel, and registers the chat for event routing.

        Returns the number of successfully restored engines.
        """
        from .models import SlockChannel

        marker_dirs = [os.path.join(self._storage_base_path, "groups")]
        existing_dirs = [path for path in marker_dirs if os.path.isdir(path)]
        if not existing_dirs:
            return 0

        restored = 0
        for slock_dir in existing_dirs:
            try:
                entries = os.listdir(slock_dir)
            except OSError as e:
                logger.warning("restore_from_disk: cannot list %s: %s", slock_dir, repr(e))
                continue

            for entry in entries:
                channel_dir = os.path.join(slock_dir, entry)
                if (
                    not _LARK_CHAT_ID_RE.fullmatch(entry)
                    or not os.path.isdir(channel_dir)
                    or os.path.islink(channel_dir)
                ):
                    continue

                marker_path = os.path.join(channel_dir, ".slock_channel.json")
                if not os.path.isfile(marker_path) or os.path.islink(marker_path):
                    continue

                try:
                    with open(marker_path, "r", encoding="utf-8") as f:
                        marker = json.load(f)
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(
                        "restore_from_disk: skipping corrupted marker %s: %s",
                        marker_path, e,
                    )
                    continue

                channel_id = marker.get("channel_id")
                if not channel_id:
                    logger.warning(
                        "restore_from_disk: marker missing channel_id: %s", marker_path
                    )
                    continue
                if (
                    not isinstance(channel_id, str)
                    or not _LARK_CHAT_ID_RE.fullmatch(channel_id)
                    or channel_id != entry
                ):
                    logger.warning(
                        "restore_from_disk: skipping non-Lark chat marker: %s",
                        marker_path,
                    )
                    continue

                if (
                    managed_group_active is not None
                    and not managed_group_active(channel_id)
                ):
                    try:
                        self.prepare_revoke_marker(channel_id)
                    except (OSError, ValueError):
                        logger.exception(
                            "restore_from_disk: cannot retire untrusted marker %s",
                            marker_path,
                        )
                    continue

                # Skip if already managed (idempotent)
                if self.is_managed_chat(channel_id):
                    continue

                try:
                    engine_root_path = self._resolve_marker_root_path(
                        marker,
                        channel_dir,
                    )
                    channel = SlockChannel(
                        channel_id=channel_id,
                        name=marker.get("name", ""),
                        team_name=marker.get("team_name", ""),
                        owner_id=marker.get("owner_id", ""),
                    )
                    self.get_or_create_activated(
                        channel_id,
                        engine_root_path,
                        channel,
                        engine_name="Slock",
                    )
                    restored += 1
                    logger.info(
                        "restore_from_disk: restored slock engine for chat=%s team=%s",
                        channel_id, channel.team_name,
                    )
                except Exception as e:
                    logger.error(
                        "restore_from_disk: failed to restore %s: %s",
                        channel_id, e,
                    )

        if restored:
            logger.info("restore_from_disk: restored %d slock engine(s)", restored)
        return restored

    @staticmethod
    def _resolve_marker_root_path(marker: dict, channel_dir: str) -> str:
        persisted_candidates = [marker.get("root_path")]
        config_path = os.path.join(channel_dir, "workspace", ".team-config.json")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                team_config = json.load(f)
            persisted_candidates.append(team_config.get("project_path"))
        except (FileNotFoundError, json.JSONDecodeError, OSError, AttributeError):
            pass
        for candidate in persisted_candidates:
            if isinstance(candidate, str) and os.path.isabs(candidate) and os.path.isdir(candidate):
                return os.path.realpath(candidate)
        raise ValueError("persisted Slock project root is unavailable")

    def _build_snapshot(self, engine: SlockEngine):
        """Build EngineSnapshot with Slock-specific fields."""
        from src.card.engine_snapshot import EngineSnapshot

        channel = engine.channel
        agents = engine.registry.list_agents(
            channel_id=channel.channel_id if channel else None
        )
        return EngineSnapshot(
            engine_name=engine.engine_name,
            root_path=engine.root_path,
            is_running=engine.is_running,
            ext={
                "channel": channel,
                "agent_count": len(agents),
                "team_name": channel.team_name if channel else "",
            },
        )
