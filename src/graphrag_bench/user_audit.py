"""Attributable user audit logging module for Graph RAG Benchmark.

Provides structured, append-only, thread-safe JSONL audit logging for user
actions, access control events, authentication events, and administrative changes.
Ensures zero secret or credential leakage with automated payload sanitization.
"""

from __future__ import annotations

import contextlib
import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from graphrag_bench.config import Settings, get_settings
from graphrag_bench.logger import get_logger

logger = get_logger(__name__)

# Sensitive key names that must be scrubbed from details
SENSITIVE_KEYS = {
    "password",
    "token",
    "secret",
    "api_key",
    "credential",
    "authorization",
    "bearer",
    "access_token",
    "id_token",
    "refresh_token",
    "cookie",
}


def sanitize_audit_details(data: Any) -> Any:
    """Recursively scrub sensitive keys and credential patterns from audit payload."""
    if isinstance(data, dict):
        sanitized = {}
        for k, v in data.items():
            if any(s in k.lower() for s in SENSITIVE_KEYS):
                sanitized[k] = "[REDACTED]"
            else:
                sanitized[k] = sanitize_audit_details(v)
        return sanitized
    elif isinstance(data, list):
        return [sanitize_audit_details(item) for item in data]
    return data


class UserAuditEvent(BaseModel):
    """Structured audit log event model."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    user_id: str
    email: str
    role: str
    action: str
    status: str = "SUCCESS"  # SUCCESS, FAILURE, DENIED
    details: dict[str, Any] = Field(default_factory=dict)
    ip_address: str | None = None
    user_agent: str | None = None

    def to_jsonl_line(self) -> str:
        """Serialize event to a single JSON line."""
        data = self.model_dump(mode="json")
        # Ensure timestamp is ISO 8601 formatted string
        if isinstance(data.get("timestamp"), datetime):
            data["timestamp"] = data["timestamp"].isoformat()
        return json.dumps(data, ensure_ascii=False)


class UserAuditLogger:
    """Thread-safe append-only user audit logger writing to JSONL."""

    def __init__(
        self,
        log_path: Path | str | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.log_path = Path(log_path or self.settings.user_audit_log_path)
        self._lock = threading.Lock()
        self._ensure_log_dir()

    def _ensure_log_dir(self) -> None:
        """Ensure parent directory exists."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        user_id: str,
        email: str,
        role: str,
        action: str,
        status: str = "SUCCESS",
        details: dict[str, Any] | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> UserAuditEvent:
        """Create, sanitize, persist, and return an audit event."""
        clean_details = sanitize_audit_details(details or {})
        event = UserAuditEvent(
            user_id=user_id,
            email=email,
            role=role,
            action=action,
            status=status,
            details=clean_details,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        line = event.to_jsonl_line() + "\n"

        with self._lock:
            try:
                self._ensure_log_dir()
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(line)
            except Exception as e:
                logger.error(f"Failed to write user audit event {event.event_id}: {e}")

        return event

    def get_recent_events(
        self,
        limit: int = 100,
        action: str | None = None,
        user_id: str | None = None,
        status: str | None = None,
    ) -> list[UserAuditEvent]:
        """Read and return recent audit events matching criteria, latest first."""
        if not self.log_path.exists():
            return []

        events: list[UserAuditEvent] = []
        with self._lock:
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except Exception as e:
                logger.error(f"Failed to read user audit log: {e}")
                return []

        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                evt = UserAuditEvent(**data)
                if action and evt.action != action:
                    continue
                if user_id and evt.user_id != user_id:
                    continue
                if status and evt.status != status:
                    continue
                events.append(evt)
                if len(events) >= limit:
                    break
            except Exception as e:
                logger.warning(f"Skipping malformed audit log entry: {e}")

        return events

    def get_summary_statistics(self) -> dict[str, Any]:
        """Calculate summary statistics over the audit log."""
        if not self.log_path.exists():
            return {
                "total_events": 0,
                "actions": {},
                "statuses": {},
                "users": {},
            }

        total = 0
        actions: dict[str, int] = {}
        statuses: dict[str, int] = {}
        users: dict[str, int] = {}

        with self._lock:
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        total += 1
                        with contextlib.suppress(Exception):
                            data = json.loads(line)
                            act = data.get("action", "unknown")
                            st = data.get("status", "unknown")
                            em = data.get("email", "unknown")

                            actions[act] = actions.get(act, 0) + 1
                            statuses[st] = statuses.get(st, 0) + 1
                            users[em] = users.get(em, 0) + 1
            except Exception as e:
                logger.error(f"Failed to calculate audit statistics: {e}")

        return {
            "total_events": total,
            "actions": actions,
            "statuses": statuses,
            "users": users,
        }
