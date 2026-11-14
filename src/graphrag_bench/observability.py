"""Observability, structured logging, and telemetry subsystem for Graph RAG Benchmark.

Provides:
1. Structured JSON and standard rotating file logging.
2. In-memory operational telemetry (latencies, counts, failure rates).
3. Thread-safe concurrency semaphores and backpressure limiters.
4. Operation tracing context managers with correlation IDs.
"""

import json
import logging
import logging.handlers
import threading
import time
import uuid
from collections import deque
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from graphrag_bench.config import Settings, get_settings


class JsonLogFormatter(logging.Formatter):
    """Formats log records as structured JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "line": record.lineno,
            "function": record.funcName,
        }
        if hasattr(record, "operation_id"):
            log_data["operation_id"] = record.operation_id
        if hasattr(record, "event_type"):
            log_data["event_type"] = record.event_type
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms
        if hasattr(record, "metadata") and isinstance(record.metadata, dict):
            log_data["metadata"] = record.metadata
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)


class TelemetryEvent:
    """Represents a recorded operational event."""

    def __init__(
        self,
        event_type: str,
        duration_ms: float,
        success: bool,
        operation_id: str,
        metadata: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        self.event_type = event_type
        self.duration_ms = duration_ms
        self.success = success
        self.operation_id = operation_id
        self.metadata = metadata or {}
        self.timestamp = timestamp or datetime.now(tz=UTC)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "duration_ms": round(self.duration_ms, 2),
            "success": self.success,
            "operation_id": self.operation_id,
            "metadata": self.metadata,
        }


class TelemetryCollector:
    """Thread-safe collector for real-time application metrics and telemetry."""

    _instance: "TelemetryCollector | None" = None
    _lock = threading.Lock()

    def __init__(self, max_history: int = 500) -> None:
        self._history: deque[TelemetryEvent] = deque(maxlen=max_history)
        self._event_lock = threading.Lock()
        self._app_start_time = time.time()
        self._active_generations = 0
        self._active_benchmarks = 0

    @classmethod
    def get_instance(cls) -> "TelemetryCollector":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._app_start_time

    def record_event(
        self,
        event_type: str,
        duration_ms: float,
        success: bool,
        operation_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record an operational telemetry event."""
        event = TelemetryEvent(
            event_type=event_type,
            duration_ms=duration_ms,
            success=success,
            operation_id=operation_id,
            metadata=metadata,
        )
        with self._event_lock:
            self._history.append(event)

    def get_recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._event_lock:
            events = list(self._history)[-limit:]
            return [e.to_dict() for e in reversed(events)]

    def get_aggregated_metrics(self) -> dict[str, dict[str, Any]]:
        """Compute aggregated metrics grouped by event_type."""
        with self._event_lock:
            events_by_type: dict[str, list[TelemetryEvent]] = {}
            for e in self._history:
                events_by_type.setdefault(e.event_type, []).append(e)

        summary: dict[str, dict[str, Any]] = {}
        for event_type, events in events_by_type.items():
            durations = [e.duration_ms for e in events]
            total_count = len(events)
            success_count = sum(1 for e in events if e.success)
            error_count = total_count - success_count
            avg_duration = sum(durations) / total_count if total_count > 0 else 0.0
            p95_duration = sorted(durations)[int(0.95 * total_count)] if total_count > 0 else 0.0

            summary[event_type] = {
                "count": total_count,
                "success_count": success_count,
                "error_count": error_count,
                "avg_duration_ms": round(avg_duration, 2),
                "p95_duration_ms": round(p95_duration, 2),
                "min_duration_ms": round(min(durations), 2) if durations else 0.0,
                "max_duration_ms": round(max(durations), 2) if durations else 0.0,
            }
        return summary


class ConcurrencyLimiter:
    """Thread-safe backpressure management for GPU inference and benchmarks."""

    _instance: "ConcurrencyLimiter | None" = None
    _init_lock = threading.Lock()

    def __init__(self, settings: Settings | None = None) -> None:
        cfg = settings or get_settings()
        self.gen_semaphore = threading.BoundedSemaphore(cfg.max_concurrent_generations)
        self.bench_semaphore = threading.BoundedSemaphore(cfg.max_concurrent_benchmarks)
        self._max_gen = cfg.max_concurrent_generations
        self._max_bench = cfg.max_concurrent_benchmarks
        self._active_gen = 0
        self._active_bench = 0
        self._count_lock = threading.Lock()

    @classmethod
    def get_instance(cls, settings: Settings | None = None) -> "ConcurrencyLimiter":
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = cls(settings)
            return cls._instance

    @contextmanager
    def acquire_generation_slot(self, timeout: float = 30.0) -> Generator[bool, None, None]:
        """Acquire a slot for LLM generation with timeout."""
        acquired = self.gen_semaphore.acquire(timeout=timeout)
        if not acquired:
            raise TimeoutError(
                f"Generation queue full: max concurrent limit ({self._max_gen}) reached. "
                "Please retry in a moment."
            )
        with self._count_lock:
            self._active_gen += 1
        try:
            yield True
        finally:
            with self._count_lock:
                self._active_gen = max(0, self._active_gen - 1)
            self.gen_semaphore.release()

    @contextmanager
    def acquire_benchmark_slot(self, timeout: float = 5.0) -> Generator[bool, None, None]:
        """Acquire an exclusive or bounded slot for benchmark runs."""
        acquired = self.bench_semaphore.acquire(timeout=timeout)
        if not acquired:
            raise RuntimeError(
                "A benchmark suite is already currently running. Concurrency limit enforced."
            )
        with self._count_lock:
            self._active_bench += 1
        try:
            yield True
        finally:
            with self._count_lock:
                self._active_bench = max(0, self._active_bench - 1)
            self.bench_semaphore.release()

    @property
    def active_generations(self) -> int:
        with self._count_lock:
            return self._active_gen

    @property
    def active_benchmarks(self) -> int:
        with self._count_lock:
            return self._active_bench


def configure_observability(settings: Settings | None = None) -> None:
    """Initialize structured rotating file logging and telemetry handlers."""
    cfg = settings or get_settings()
    log_level = getattr(logging, cfg.log_level.upper(), logging.INFO)
    pkg_logger = logging.getLogger("graphrag_bench")
    pkg_logger.setLevel(log_level)

    log_file_path = Path(cfg.log_file).resolve()
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Standard Rotating File Handler
    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_file_path),
        maxBytes=cfg.log_max_bytes,
        backupCount=cfg.log_backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)

    if cfg.log_format == "json":
        file_handler.setFormatter(JsonLogFormatter())
    else:
        text_fmt = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
        file_handler.setFormatter(logging.Formatter(fmt=text_fmt, datefmt="%Y-%m-%d %H:%M:%S"))

    # Also generate a companion jsonl file for structured log ingestion
    json_log_path = log_file_path.parent / f"{log_file_path.stem}.jsonl"
    json_handler = logging.handlers.RotatingFileHandler(
        filename=str(json_log_path),
        maxBytes=cfg.log_max_bytes,
        backupCount=cfg.log_backup_count,
        encoding="utf-8",
    )
    json_handler.setLevel(log_level)
    json_handler.setFormatter(JsonLogFormatter())

    # Avoid duplicate handlers on re-call
    existing_types = [type(h) for h in pkg_logger.handlers]
    if logging.handlers.RotatingFileHandler not in existing_types:
        pkg_logger.addHandler(file_handler)
        pkg_logger.addHandler(json_handler)


@contextmanager
def trace_operation(
    event_type: str,
    operation_id: str | None = None,
    logger: logging.Logger | None = None,
    metadata: dict[str, Any] | None = None,
) -> Generator[str, None, None]:
    """Context manager to trace execution duration, log structured event, and update telemetry."""
    op_id = operation_id or f"op_{uuid.uuid4().hex[:12]}"
    meta = metadata or {}
    log = logger or logging.getLogger("graphrag_bench.observability")
    telemetry = TelemetryCollector.get_instance()

    start_time = time.perf_counter()
    try:
        yield op_id
    except Exception as exc:
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        telemetry.record_event(
            event_type=event_type,
            duration_ms=duration_ms,
            success=False,
            operation_id=op_id,
            metadata={**meta, "error": str(exc)},
        )
        log.error(
            "Operation %s (%s) failed after %.2f ms: %s",
            op_id,
            event_type,
            duration_ms,
            exc,
            extra={
                "operation_id": op_id,
                "event_type": event_type,
                "duration_ms": duration_ms,
                "metadata": meta,
            },
        )
        raise
    else:
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        telemetry.record_event(
            event_type=event_type,
            duration_ms=duration_ms,
            success=True,
            operation_id=op_id,
            metadata=meta,
        )
        log.info(
            "Operation %s (%s) completed in %.2f ms",
            op_id,
            event_type,
            duration_ms,
            extra={
                "operation_id": op_id,
                "event_type": event_type,
                "duration_ms": duration_ms,
                "metadata": meta,
            },
        )
