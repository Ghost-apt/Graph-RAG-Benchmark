"""Unit tests for Phase 9 structured logging, telemetry, and concurrency limiting."""

import json
import logging
import time

from graphrag_bench.observability import (
    ConcurrencyLimiter,
    JsonLogFormatter,
    TelemetryCollector,
    trace_operation,
)


class TestObservability:
    """Test suite for structured observability and telemetry subsystem."""

    def test_json_log_formatter(self):
        """Verify JsonLogFormatter produces valid structured JSON lines."""
        formatter = JsonLogFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname=__file__,
            lineno=42,
            msg="User %s logged in",
            args=("admin",),
            exc_info=None,
            func="test_func",
        )
        record.operation_id = "op_12345"
        record.event_type = "auth_event"
        record.duration_ms = 45.2
        record.metadata = {"client_ip": "127.0.0.1"}

        formatted = formatter.format(record)
        data = json.loads(formatted)
        assert data["level"] == "INFO"
        assert data["logger"] == "test_logger"
        assert data["message"] == "User admin logged in"
        assert data["operation_id"] == "op_12345"
        assert data["event_type"] == "auth_event"
        assert data["duration_ms"] == 45.2
        assert data["metadata"]["client_ip"] == "127.0.0.1"

    def test_telemetry_collector_recording_and_aggregation(self):
        """Verify TelemetryCollector records events and calculates stats."""
        telemetry = TelemetryCollector(max_history=50)
        telemetry.record_event(
            event_type="test_retrieval",
            duration_ms=10.0,
            success=True,
            operation_id="op_1",
        )
        telemetry.record_event(
            event_type="test_retrieval",
            duration_ms=20.0,
            success=True,
            operation_id="op_2",
        )
        telemetry.record_event(
            event_type="test_retrieval",
            duration_ms=30.0,
            success=False,
            operation_id="op_3",
        )

        recent = telemetry.get_recent_events(limit=5)
        assert len(recent) == 3
        assert recent[0]["operation_id"] == "op_3"  # newest first

        summary = telemetry.get_aggregated_metrics()
        assert "test_retrieval" in summary
        stats = summary["test_retrieval"]
        assert stats["count"] == 3
        assert stats["success_count"] == 2
        assert stats["error_count"] == 1
        assert stats["avg_duration_ms"] == 20.0
        assert stats["min_duration_ms"] == 10.0
        assert stats["max_duration_ms"] == 30.0

    def test_concurrency_limiter_generation_slots(self):
        """Verify ConcurrencyLimiter enforces bounded generation semaphore."""
        limiter = ConcurrencyLimiter()
        assert limiter.active_generations == 0

        with limiter.acquire_generation_slot():
            assert limiter.active_generations == 1
        assert limiter.active_generations == 0

    def test_trace_operation_context_manager(self):
        """Verify trace_operation records telemetry and returns operation ID."""
        with trace_operation("sample_op", metadata={"param": 1}) as op_id:
            assert op_id.startswith("op_")
            time.sleep(0.01)

        telemetry = TelemetryCollector.get_instance()
        recent = telemetry.get_recent_events(limit=1)
        assert len(recent) > 0
        assert recent[0]["event_type"] == "sample_op"
        assert recent[0]["success"] is True
        assert recent[0]["duration_ms"] > 5.0
