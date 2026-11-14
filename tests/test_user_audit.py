"""Unit tests for Phase 10 User Audit Logging.

Tests:
- UserAuditEvent schema validation and JSONL serialization
- Thread-safe append and retrieval
- Zero-leakage credential redaction
- Audit filtering and summary calculations
"""

from __future__ import annotations

from pathlib import Path

from graphrag_bench.user_audit import (
    UserAuditEvent,
    UserAuditLogger,
    sanitize_audit_details,
)


class TestUserAudit:
    """Audit logging validation suite."""

    def test_audit_event_model(self) -> None:
        """Verify UserAuditEvent model fields and JSONL string output."""
        event = UserAuditEvent(
            user_id="usr_test1",
            email="researcher@test.com",
            role="researcher",
            action="QUERY_EXECUTION",
            status="SUCCESS",
            details={"mode": "hybrid", "top_k": 5},
        )
        assert event.event_id is not None
        assert event.timestamp is not None
        line = event.to_jsonl_line()
        assert "researcher@test.com" in line
        assert "QUERY_EXECUTION" in line
        assert "hybrid" in line

    def test_credential_sanitization(self) -> None:
        """Verify recursive scrubbing of credentials and sensitive keys."""
        raw_payload = {
            "query": "What is Graph RAG?",
            "api_key": "sk-1234567890abcdef",
            "access_token": "Bearer eyJhbGciOi...",
            "nested": {
                "password": "supersecretpassword",
                "client_secret": "cs_test_9876",
                "safe_field": 42,
            },
            "array_items": [
                {"token": "xyz", "name": "item1"},
                {"authorization": "basic auth", "id": 101},
            ],
        }

        clean = sanitize_audit_details(raw_payload)
        assert clean["query"] == "What is Graph RAG?"
        assert clean["api_key"] == "[REDACTED]"
        assert clean["access_token"] == "[REDACTED]"
        assert clean["nested"]["password"] == "[REDACTED]"
        assert clean["nested"]["client_secret"] == "[REDACTED]"
        assert clean["nested"]["safe_field"] == 42
        assert clean["array_items"][0]["token"] == "[REDACTED]"
        assert clean["array_items"][0]["name"] == "item1"
        assert clean["array_items"][1]["authorization"] == "[REDACTED]"
        assert clean["array_items"][1]["id"] == 101

    def test_audit_logger_write_and_read(self, tmp_path: Path) -> None:
        """Verify writing and reading back events, with latest-first ordering."""
        log_file = tmp_path / "user_audit.jsonl"
        logger = UserAuditLogger(log_path=log_file)

        e1 = logger.log_event(
            user_id="usr_1",
            email="u1@test.com",
            role="admin",
            action="AUTH_LOGIN",
            status="SUCCESS",
        )
        e2 = logger.log_event(
            user_id="usr_2",
            email="u2@test.com",
            role="researcher",
            action="QUERY_EXECUTION",
            status="SUCCESS",
        )
        e3 = logger.log_event(
            user_id="usr_3",
            email="u3@test.com",
            role="viewer",
            action="DOCUMENTS_UPLOAD",
            status="DENIED",
        )

        events = logger.get_recent_events(limit=10)
        assert len(events) == 3
        # Ordering must be latest first
        assert events[0].event_id == e3.event_id
        assert events[1].event_id == e2.event_id
        assert events[2].event_id == e1.event_id

    def test_audit_logger_filtering(self, tmp_path: Path) -> None:
        """Verify filtering by action, status, and user_id."""
        log_file = tmp_path / "user_audit.jsonl"
        logger = UserAuditLogger(log_path=log_file)

        logger.log_event("u1", "u1@test.com", "admin", "AUTH_LOGIN", "SUCCESS")
        logger.log_event("u1", "u1@test.com", "admin", "QUERY_EXECUTION", "SUCCESS")
        logger.log_event("u2", "u2@test.com", "viewer", "AUTH_LOGIN", "SUCCESS")
        logger.log_event("u2", "u2@test.com", "viewer", "DOCUMENTS_UPLOAD", "DENIED")

        # Filter by action
        logins = logger.get_recent_events(action="AUTH_LOGIN")
        assert len(logins) == 2

        # Filter by status
        denied = logger.get_recent_events(status="DENIED")
        assert len(denied) == 1
        assert denied[0].user_id == "u2"

        # Filter by user_id
        u1_events = logger.get_recent_events(user_id="u1")
        assert len(u1_events) == 2

    def test_summary_statistics(self, tmp_path: Path) -> None:
        """Verify aggregation counts."""
        log_file = tmp_path / "user_audit.jsonl"
        logger = UserAuditLogger(log_path=log_file)

        logger.log_event("u1", "u1@test.com", "admin", "AUTH_LOGIN", "SUCCESS")
        logger.log_event("u2", "u2@test.com", "researcher", "AUTH_LOGIN", "SUCCESS")
        logger.log_event("u2", "u2@test.com", "researcher", "QUERY_EXECUTION", "SUCCESS")

        stats = logger.get_summary_statistics()
        assert stats["total_events"] == 3
        assert stats["actions"]["AUTH_LOGIN"] == 2
        assert stats["actions"]["QUERY_EXECUTION"] == 1
        assert stats["statuses"]["SUCCESS"] == 3
        assert stats["users"]["u1@test.com"] == 1
        assert stats["users"]["u2@test.com"] == 2

    def test_malformed_lines_handled_gracefully(self, tmp_path: Path) -> None:
        """Verify corrupted JSON lines do not crash audit reader."""
        log_file = tmp_path / "user_audit.jsonl"
        logger = UserAuditLogger(log_path=log_file)

        logger.log_event("u1", "u1@test.com", "admin", "AUTH_LOGIN", "SUCCESS")

        # Inject corrupted line
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("CORRUPTED_JSON_NOT_VALID_LINE\n")

        logger.log_event("u2", "u2@test.com", "viewer", "AUTH_LOGOUT", "SUCCESS")

        events = logger.get_recent_events(limit=10)
        assert len(events) == 2
        assert events[0].action == "AUTH_LOGOUT"
        assert events[1].action == "AUTH_LOGIN"
