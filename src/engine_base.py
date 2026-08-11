import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Generic, Optional, Protocol, TypeVar, runtime_checkable

from .acp import ACPEventRenderer
from .agent_session import SyncSession, close_session_safely
from .config import get_settings
from .utils.engine_identity import resolve_engine_identity
from .utils.errors import get_error_detail
from .utils.gc_monitor import get_gc_monitor
from .utils.lock_order import LockLevel, ordered_lock, ordered_rlock

logger = logging.getLogger(__name__)


class EngineRunState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"


class ReviewPerspective(Enum):
    ARCHITECT = "architect"
    PRODUCT = "product"
    USER = "user"
    TESTER = "tester"
    DESIGNER = "designer"

    @classmethod
    def register_display_names(cls, mapping: dict[str, str]) -> None:
        """Inject display-name mapping (called by spec_engine on init).

        *mapping* keys are ``"perspective_<member>"`` strings, values are
        localised display names.  This keeps the dependency direction
        spec_engine → engine_base (not the reverse).
        """
        cls._display_names.update(mapping)

    @property
    def display_name(self) -> str:
        _key_map = {
            ReviewPerspective.ARCHITECT: "perspective_architect",
            ReviewPerspective.PRODUCT: "perspective_product",
            ReviewPerspective.USER: "perspective_user",
            ReviewPerspective.TESTER: "perspective_tester",
            ReviewPerspective.DESIGNER: "perspective_designer",
        }
        return self._display_names.get(_key_map[self], self.value)

    @property
    def emoji(self) -> str:
        return {
            ReviewPerspective.ARCHITECT: "🏗️",
            ReviewPerspective.PRODUCT: "📋",
            ReviewPerspective.USER: "👤",
            ReviewPerspective.TESTER: "🧪",
            ReviewPerspective.DESIGNER: "🎨",
        }[self]

    @property
    def review_focus(self) -> str:
        return {
            ReviewPerspective.ARCHITECT: "代码结构、设计模式、可维护性、性能、安全性",
            ReviewPerspective.PRODUCT: "需求完整度、用户价值、边界场景、功能一致性",
            ReviewPerspective.USER: "易用性、文档、错误提示、交互体验、可理解性",
            ReviewPerspective.TESTER: "测试覆盖、边界条件、异常处理、回归风险、可测试性",
            ReviewPerspective.DESIGNER: "UI视觉(配色/层级)、交互体验(动效/流程)、移动端适配、美观度",
        }[self]

    @property
    def failure_label(self) -> str:
        return {
            ReviewPerspective.DESIGNER: "🎨 视觉/交互建议",
        }.get(self, "❌ 有建议")


# Storage for injected display names — kept outside the Enum body to avoid
# creating an enum member.  Accessed as ReviewPerspective._display_names.
ReviewPerspective._display_names = {}  # type: ignore[attr-defined]


@dataclass
class PerspectiveReview:
    perspective: ReviewPerspective
    passed: bool
    suggestions: list[str] = field(default_factory=list)
    summary: str = ""
    role_id: str = ""
    role_display_name: str = ""
    role_category: str = ""
    review_agent_label: str = ""
    review_agent_type: str = ""
    review_model_name: str = ""
    blocking: bool = True

    def to_dict(self) -> dict:
        return {
            "perspective": self.perspective.value,
            "passed": self.passed,
            "suggestions": self.suggestions,
            "summary": self.summary,
            "role_id": self.role_id,
            "role_display_name": self.role_display_name,
            "role_category": self.role_category,
            "review_agent_label": self.review_agent_label,
            "review_agent_type": self.review_agent_type,
            "review_model_name": self.review_model_name,
            "blocking": self.blocking,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PerspectiveReview":
        return cls(
            perspective=ReviewPerspective(data["perspective"]),
            passed=data["passed"],
            suggestions=data.get("suggestions", []),
            summary=data.get("summary", ""),
            role_id=str(data.get("role_id") or ""),
            role_display_name=str(data.get("role_display_name") or ""),
            role_category=str(data.get("role_category") or ""),
            review_agent_label=str(data.get("review_agent_label") or ""),
            review_agent_type=str(data.get("review_agent_type") or ""),
            review_model_name=str(data.get("review_model_name") or ""),
            blocking=bool(data.get("blocking", True)),
        )


@dataclass
class ReviewResult:
    reviews: list[PerspectiveReview] = field(default_factory=list)
    iteration: int = 0

    @property
    def all_passed(self) -> bool:
        return len(self.reviews) > 0 and all(r.passed for r in self.reviews)

    @property
    def total_suggestions(self) -> int:
        return sum(len(r.suggestions) for r in self.reviews)

    @property
    def failed_perspectives(self) -> list[PerspectiveReview]:
        return [r for r in self.reviews if not r.passed]

    def suggestions_by_perspective(self) -> dict[ReviewPerspective, list[str]]:
        return {r.perspective: r.suggestions for r in self.reviews if r.suggestions}

    def to_dict(self) -> dict:
        return {
            "reviews": [r.to_dict() for r in self.reviews],
            "iteration": self.iteration,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReviewResult":
        return cls(
            reviews=[PerspectiveReview.from_dict(r) for r in data.get("reviews", [])],
            iteration=data.get("iteration", 0),
        )


@runtime_checkable
class HasOnError(Protocol):
    """Protocol for callback objects that may carry an ``on_error`` handler."""

    on_error: Optional[Callable[[str], None]]


T = TypeVar("T", bound="BaseEngine")


class BaseEngine:

    _state_filename: str = ".engine_state.json"
    _gc_label: str = "Engine"
    _gc_threshold_default: float = 85.0

    def __init__(
        self,
        chat_id: str,
        root_path: str,
        agent_type: str = "coco",
        engine_name: str = "Coco",
        model_name: Optional[str] = None,
        settings: Optional[object] = None,
    ):
        self.chat_id = chat_id
        self.root_path = os.path.expanduser(root_path)
        self.settings = settings if settings is not None else get_settings()
        self.engine_name = engine_name
        self._agent_type = agent_type
        self._model_name = model_name

        self._session: Optional[SyncSession] = None
        self._project = None
        self._renderer = ACPEventRenderer()
        self._run_state = EngineRunState.IDLE
        self._lock = ordered_rlock(LockLevel.ENGINE_INSTANCE, name="BaseEngine._lock")

    @property
    def project(self):
        with self._lock:
            return self._project

    @property
    def run_state(self) -> EngineRunState:
        with self._lock:
            return self._run_state

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._run_state != EngineRunState.IDLE

    def _close_session_safely(self) -> None:
        close_session_safely(self._session)
        self._session = None

    def _on_stop(self) -> None:
        """Hook for subclasses to execute custom stop logic (e.g. cancel events).

        Called at the end of :meth:`stop`. Override in subclasses instead of
        relying on duck-typed attributes.  Default implementation is a no-op.
        """

    def stop(self):
        with self._lock:
            self._run_state = EngineRunState.STOPPING
            session = self._session
        if session:
            try:
                session.cancel()
            except Exception:
                logger.debug("session.cancel() failed during stop", exc_info=True)
        self._on_stop()

    def cleanup(self):
        # Snapshot under lock (mirrors stop() pattern to avoid TOCTOU with concurrent stop())
        with self._lock:
            state = self._run_state
            session = self._session
            if state != EngineRunState.IDLE:
                self._run_state = EngineRunState.STOPPING
            else:
                # IDLE path: clear references under lock, close session outside
                self._session = None
                self._project = None

        if state != EngineRunState.IDLE:
            if session:
                try:
                    session.cancel()
                except Exception:
                    logger.debug("session.cancel() failed during cleanup", exc_info=True)
            self._on_stop()
            return

        # IDLE path: close session and run GC outside lock
        if session:
            try:
                session.close()
            except Exception as e:
                logger.debug("关闭ACP session失败: %s", get_error_detail(e))
        mem_snapshot = getattr(self, "_mem_snapshot", None)
        gc_kwargs: dict = {"label": self._gc_label}
        if mem_snapshot is not None:
            gc_kwargs["mem_snapshot"] = mem_snapshot
        get_gc_monitor(
            memory_threshold_percent=self._gc_threshold_default,
        ).check_and_collect(**gc_kwargs)

    def save_state(self, filepath: Optional[str] = None) -> str:
        if not self._project:
            raise ValueError("没有项目状态可保存")
        if not filepath:
            filepath = os.path.join(self.root_path, self._state_filename)
        state = {
            "chat_id": self.chat_id,
            "root_path": self.root_path,
            "project": self._project.to_dict(),
            "saved_at": time.time(),
        }
        tmp_path = filepath + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, filepath)
        return filepath

    def inject_guidance(self, text: str) -> None:
        self._pending_guidance = text

    def _format_engine_error(
        self,
        error: Exception,
        label: str,
        *,
        is_timeout: bool = False,
        callbacks: Optional[HasOnError] = None,
    ) -> str:
        """Shared error formatting: build message, log, fire callback.

        Args:
            error: The caught exception.
            label: Human-readable prefix (e.g. "Spec执行", "Deep执行").
            is_timeout: True for TimeoutError (warning), False for generic (error).
            callbacks: An object with an optional ``on_error`` callable attribute.

        Returns:
            The formatted user-facing error message.
        """
        from .card.ui_text import UI_TEXT

        detail = get_error_detail(error)
        kind = UI_TEXT.get("engine_error_timeout", "超时") if is_timeout else UI_TEXT.get("engine_error_exception", "异常")
        error_msg = f"{label}{kind}: {detail}"
        project_name = getattr(self._project, "name", None) or "unknown"
        log_fn = logger.warning if is_timeout else logger.error
        log_fn("[%s:%s] %s", self.engine_name, project_name, error_msg)
        if callbacks is not None and callbacks.on_error is not None:
            callbacks.on_error(error_msg)
        return error_msg

    def get_rendered_content(self) -> str:
        return self._renderer.get_final_content()


class BaseEngineManager(Generic[T]):

    def __init__(self):
        self._engines: dict[str, T] = {}
        self._chat_keys: dict[str, set[str]] = {}
        self._lock = ordered_lock(LockLevel.ENGINE_MANAGER, name="BaseEngineManager._lock")

    def _add_index(self, chat_id: str, key: str) -> None:
        self._chat_keys.setdefault(chat_id, set()).add(key)

    def _remove_index(self, chat_id: str, key: str) -> None:
        keys = self._chat_keys.get(chat_id)
        if keys:
            keys.discard(key)
            if not keys:
                del self._chat_keys[chat_id]

    def _iter_chat_engines(self, chat_id: str):
        # Snapshot keys under lock to avoid RuntimeError: dictionary changed size during iteration
        with self._lock:
            keys = list(self._chat_keys.get(chat_id, ()))
        for key in keys:
            engine = self._engines.get(key)
            if engine:
                yield engine

    def _resolve_identity(self, engine_name: str) -> tuple[str, str, Optional[str]]:
        from .mode import InteractionMode
        normalized = (engine_name or "").strip().lower()
        if normalized.startswith("claude"):
            mode = InteractionMode.CLAUDE
        elif normalized.startswith("aiden"):
            mode = InteractionMode.AIDEN
        elif normalized.startswith("codex"):
            mode = InteractionMode.CODEX
        elif normalized.startswith("gemini"):
            mode = InteractionMode.GEMINI
        elif normalized.startswith("traex"):
            mode = InteractionMode.TRAEX
        elif normalized.startswith("grok"):
            mode = InteractionMode.GROK
        else:
            mode = InteractionMode.COCO

        identity = resolve_engine_identity(mode=mode)
        return identity.engine_name, identity.agent_type, identity.model_name

    def _create_engine(
        self,
        chat_id: str,
        root_path: str,
        agent_type: str,
        engine_name: str,
        model_name: Optional[str],
    ) -> T:
        raise NotImplementedError

    def get_or_create(
        self,
        chat_id: str,
        root_path: str,
        engine_name: str = "Coco",
        *,
        model_name: Optional[str] = None,
    ) -> T:
        key = f"{chat_id}:{root_path}"
        resolved_engine_name, agent_type, resolved_model_name = self._resolve_identity(engine_name)
        if model_name:
            resolved_model_name = model_name

        with self._lock:
            if key not in self._engines:
                engine = self._create_engine(
                    chat_id=chat_id,
                    root_path=root_path,
                    agent_type=agent_type,
                    engine_name=resolved_engine_name,
                    model_name=resolved_model_name,
                )
                self._engines[key] = engine
                self._add_index(chat_id, key)
            else:
                existing = self._engines[key]
                should_replace = (
                    existing.engine_name.lower() != resolved_engine_name.lower()
                    or getattr(existing, "_agent_type", None) != agent_type
                    or getattr(existing, "_model_name", None) != resolved_model_name
                )
                if should_replace and not existing.is_running:
                    cleanup_result = existing.cleanup()
                    if cleanup_result is False:
                        return existing
                    engine = self._create_engine(
                        chat_id=chat_id,
                        root_path=root_path,
                        agent_type=agent_type,
                        engine_name=resolved_engine_name,
                        model_name=resolved_model_name,
                    )
                    self._engines[key] = engine
            return self._engines[key]

    def get(self, chat_id: str, root_path: str) -> Optional[T]:
        key = f"{chat_id}:{root_path}"
        return self._engines.get(key)

    def get_active_engine(self, chat_id: str) -> Optional[T]:
        for engine in self._iter_chat_engines(chat_id):
            if engine.is_running:
                return engine
        return None

    def get_active_engines(self, chat_id: str) -> list[T]:
        return [e for e in self._iter_chat_engines(chat_id) if e.is_running]

    def snapshot(self, chat_id: str, root_path: str):
        """Return an EngineSnapshot DTO for rendering, or None if engine not found.

        Subclasses should override _build_snapshot() for engine-specific fields.
        """
        engine = self.get(chat_id, root_path)
        if engine is None:
            return None
        return self._build_snapshot(engine)

    def snapshot_active(self, chat_id: str) -> list:
        """Return snapshots of all active engines for a given chat."""
        return [self._build_snapshot(e) for e in self.get_active_engines(chat_id)]

    def _build_snapshot(self, engine: T):
        """Build an EngineSnapshot from an engine instance.

        Default implementation provides common fields. Subclasses override
        to add engine-specific data.
        """
        from src.card.engine_snapshot import EngineSnapshot
        return EngineSnapshot(
            engine_name=getattr(engine, "engine_name", ""),
            root_path=getattr(engine, "root_path", ""),
            is_running=getattr(engine, "is_running", False),
        )

    def list_engines(self, chat_id: Optional[str] = None) -> list[T]:
        if chat_id is None:
            return list(self._engines.values())
        return list(self._iter_chat_engines(chat_id))

    def cleanup_all(self):
        with self._lock:
            next_engines: dict[str, T] = {}
            for key, engine in self._engines.items():
                cleanup_result = engine.cleanup()
                if cleanup_result is False or engine.is_running:
                    next_engines[key] = engine
            self._engines = next_engines
            self._chat_keys.clear()
            for key in next_engines:
                chat_id = key.partition(":")[0]
                self._chat_keys.setdefault(chat_id, set()).add(key)


@dataclass
class CriteriaTracker:
    """验收标准追踪器。"""

    criteria: list[str] = field(default_factory=list)
    satisfied: dict[int, bool] = field(default_factory=dict)
    satisfied_at_iteration: dict[int, int] = field(default_factory=dict)

    def init_criteria(self, criteria: list[str]):
        self.criteria = criteria
        self.satisfied = dict.fromkeys(range(len(criteria)), False)
        self.satisfied_at_iteration = {}

    def update(self, criteria_id: int, is_satisfied: bool, iteration_id: int):
        if criteria_id < 0 or criteria_id >= len(self.criteria):
            return
        if is_satisfied and not self.satisfied.get(criteria_id, False):
            self.satisfied[criteria_id] = True
            self.satisfied_at_iteration[criteria_id] = iteration_id
        elif not is_satisfied and self.satisfied.get(criteria_id, False):
            # Allow re-evaluation: criteria can regress from PASS to FAIL.
            # This prevents a single optimistic evaluation from permanently
            # locking all criteria as satisfied.
            self.satisfied[criteria_id] = False
            self.satisfied_at_iteration.pop(criteria_id, None)

    def batch_update(self, progress: dict[int, bool], iteration_id: int):
        for criteria_id, is_satisfied in progress.items():
            self.update(criteria_id, is_satisfied, iteration_id)

    @property
    def satisfied_count(self) -> int:
        return sum(1 for v in self.satisfied.values() if v)

    @property
    def total_count(self) -> int:
        return len(self.criteria)

    @property
    def is_all_satisfied(self) -> bool:
        return self.total_count > 0 and self.satisfied_count == self.total_count

    @property
    def unsatisfied_indices(self) -> list[int]:
        return [i for i, v in self.satisfied.items() if not v]

    @property
    def unsatisfied_criteria(self) -> list[str]:
        return [self.criteria[i] for i in self.unsatisfied_indices]

    @property
    def satisfied_criteria(self) -> list[str]:
        return [self.criteria[i] for i, v in self.satisfied.items() if v]

    def to_dict(self) -> dict:
        return {
            "criteria": self.criteria,
            "satisfied": self.satisfied,
            "satisfied_at_iteration": self.satisfied_at_iteration,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CriteriaTracker":
        tracker = cls()
        tracker.criteria = data.get("criteria", [])
        # JSON keys are strings, convert to int
        tracker.satisfied = {int(k): v for k, v in data.get("satisfied", {}).items()}
        tracker.satisfied_at_iteration = {int(k): v for k, v in data.get("satisfied_at_iteration", {}).items()}
        return tracker
