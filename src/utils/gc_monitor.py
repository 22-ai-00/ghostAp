import gc
import logging
import os
import threading
import time
from typing import Optional

from .errors import get_error_detail

try:
    import psutil
except Exception:  # pragma: no cover - exercised via monkeypatch in tests
    psutil = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class GCMonitor:
    """System resource monitor and Garbage Collection trigger mechanism.

    This component encapsulates `psutil` and `gc` dependencies to decouple
    memory monitoring logic from the core business engines.
    """

    def __init__(self, memory_threshold_percent: float = 85.0, check_interval_seconds: float = 5.0):
        self._memory_threshold_percent = memory_threshold_percent
        self._check_interval_seconds = check_interval_seconds
        self._last_mem_check = 0.0
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

    def check_and_collect(self, label: str = "Engine", mem_snapshot=None) -> None:
        """Check memory usage and trigger garbage collection if threshold is exceeded.

        Args:
            label: Identifier for logging purposes (e.g., "[Deep]").
            mem_snapshot: Optional MemorySnapshot instance for detailed growth logging.
        """
        now = time.time()

        with self._lock:
            if now - self._last_mem_check < self._check_interval_seconds:
                return
            self._last_mem_check = now

        if psutil is None:
            logger.debug("[%s] psutil unavailable, skip memory monitoring", label)
            return

        try:
            process = psutil.Process(os.getpid())
            mem_percent = process.memory_percent()

            if mem_percent > self._memory_threshold_percent:
                logger.warning(f"[{label}] 内存告警: 使用率 {mem_percent:.1f}%, 触发主动GC")

                # Log memory growth before GC to identify potential leaks if a snapshot is provided
                if mem_snapshot is not None:
                    try:
                        mem_snapshot.log_growth(logger_func=logger.warning)
                    except Exception as ex:
                        logger.warning(f"[{label}] 内存快照分析失败: {get_error_detail(ex)}")

                # Trigger garbage collection
                gc.collect()
                try:
                    from .memory_tuning import trim_process_heap

                    trim_process_heap()
                except Exception:
                    pass

                mem_percent_after = process.memory_percent()
                logger.info(f"[{label}] GC后内存: {mem_percent_after:.1f}%")

        except Exception as e:
            logger.debug(f"[{label}] 内存监控失败: {get_error_detail(e)}")


_global_gc_monitor: Optional[GCMonitor] = None


def get_gc_monitor(memory_threshold_percent: float = 85.0) -> GCMonitor:
    """Get the global GCMonitor instance."""
    global _global_gc_monitor
    if _global_gc_monitor is None:
        _global_gc_monitor = GCMonitor(memory_threshold_percent=memory_threshold_percent)
    else:
        _global_gc_monitor._memory_threshold_percent = memory_threshold_percent
    return _global_gc_monitor
