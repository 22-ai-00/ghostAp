import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..engine_base import EngineRunState  # noqa: F401 — compatibility re-export


class DeepProjectStatus(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProgressUpdate:
    project_id: str
    completed_count: int
    total_count: int
    status: DeepProjectStatus
    message: str
    timestamp: float = field(default_factory=time.time)

    @property
    def progress_percent(self) -> float:
        if self.total_count == 0:
            return 0.0
        return (self.completed_count / self.total_count) * 100


@dataclass
class DeepProject:
    project_id: str
    name: str
    root_path: str
    status: DeepProjectStatus = DeepProjectStatus.IDLE
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    task_id: Optional[str] = None  # Human-readable task ID

    @classmethod
    def create(cls, name: str, root_path: str) -> "DeepProject":
        return cls(
            project_id=str(uuid.uuid4())[:8],
            name=name,
            root_path=root_path,
        )

    def start(self):
        self.status = DeepProjectStatus.EXECUTING
        # started_at may already be set when entering PLANNING; keep the earliest.
        if self.started_at is None:
            self.started_at = time.time()

    def cancel(self):
        self.status = DeepProjectStatus.CANCELLED
        self.completed_at = time.time()

    def complete(self):
        self.status = DeepProjectStatus.COMPLETED
        self.completed_at = time.time()
        self.error = None

    def fail(self, error: str):
        self.status = DeepProjectStatus.FAILED
        self.completed_at = time.time()
        self.error = error

    def duration(self) -> Optional[float]:
        if self.started_at:
            end = self.completed_at or time.time()
            return end - self.started_at
        return None

    def to_dict(self) -> dict:
        return {
            "project_id": self.project_id,
            "name": self.name,
            "root_path": self.root_path,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "task_id": self.task_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DeepProject":
        raw_status = data.get("status", "idle")
        if raw_status == "paused":
            raw_status = DeepProjectStatus.CANCELLED.value
        return cls(
            project_id=data["project_id"],
            name=data["name"],
            root_path=data["root_path"],
            status=DeepProjectStatus(raw_status),
            created_at=data.get("created_at", time.time()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            error=data.get("error"),
            task_id=data.get("task_id"),
        )
