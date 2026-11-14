# Graph RAG Benchmark — Phase 10 Audit Logging Reference

**Document Version:** `1.0.0`  
**Release Target:** `v0.10.0` (Phase 10 — Real-User Pilot Deployment & Product Launch)  
**Classification:** Security & Audit Specification  
**Status:** Approved & Enforced  

---

## 1. Overview & Architectural Principles

The Graph RAG Benchmark platform implements a **structured, attributable, append-only JSONL audit log** located at `logs/user_audit.jsonl` (configurable via `Settings.user_audit_log_path`).

Every action that modifies shared state, consumes GPU resources, triggers authentication handshakes, or attempts privileged operations is captured as an immutable event. The audit subsystem adheres to three non-negotiable principles:
1. **Attribution:** Every action is linked to a concrete `user_id`, `email`, and `role`.
2. **Zero Credential Leakage:** Sensitive tokens, client secrets, passwords, and authorization headers are recursively sanitized prior to disk write.
3. **Thread-Safe Append-Only Integrity:** Multiple concurrent worker threads append safely using internal locks without interleaving or corrupting JSON lines.

---

## 2. Event Schema Specification

Each event is serialized as a single, valid JSON object on its own line (`application/jsonl` format):

```json
{
  "event_id": "a9d7b42c-81e5-4c0a-93be-d14f2e5a7690",
  "timestamp": "2026-11-14T10:20:15.123456Z",
  "user_id": "usr_7a18f9c2d1b045e3",
  "email": "jane.doe@organization.com",
  "role": "researcher",
  "action": "QUERY_EXECUTION",
  "status": "SUCCESS",
  "details": {
    "retrieval_mode": "hybrid",
    "top_k": 5,
    "answer_status": "ANSWERED",
    "evidence_count": 5,
    "citations_count": 3,
    "query_length": 48
  },
  "ip_address": "192.168.1.105",
  "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)..."
}
```

### Field Definitions

| Field Name | Type | Format | Nullable | Description |
| :--- | :---: | :---: | :---: | :--- |
| `event_id` | `str` | UUIDv4 | No | Unique identifier generated per event. |
| `timestamp` | `str` | ISO 8601 UTC | No | Precise UTC timestamp of event generation. |
| `user_id` | `str` | `usr_{16hex}` | No | Deterministic user identifier from UserRegistry. |
| `email` | `str` | RFC 5322 Email | No | Normalized email of authenticated principal. |
| `role` | `str` | `admin`, `researcher`, `viewer`, `none` | No | Active RBAC role at execution time. |
| `action` | `str` | `UPPER_SNAKE_CASE` | No | Operational event type identifier. |
| `status` | `str` | `SUCCESS`, `FAILURE`, `DENIED` | No | Outcome of the operation. |
| `details` | `dict` | JSON Object | No | Contextual metadata dictionary (scrubbed). |
| `ip_address` | `str` | IPv4 / IPv6 | Yes | Client IP address if provided by reverse proxy. |
| `user_agent` | `str` | User-Agent string | Yes | Browser or client application identifier. |

---

## 3. Comprehensive Event Catalog

The platform defines 14 standard event types:

| Action Identifier | Typical Status | Triggering Condition | Details Metadata Captured |
| :--- | :---: | :--- | :--- |
| `AUTH_LOGIN` | `SUCCESS` | User completes OIDC or dev persona login. | `provider`, `mode`, `persona` |
| `AUTH_LOGOUT` | `SUCCESS` | User clicks Sign Out or terminates session. | `mode`, `session_duration` |
| `AUTH_FAILURE` | `FAILURE` / `DENIED` | Token missing email, email unverified, or not on allowlist. | `reason`, `missing_claims` |
| `AUTH_SESSION_EXPIRED` | `SUCCESS` | Inactivity exceeds `SESSION_TIMEOUT_MINUTES`. | `elapsed_minutes`, `timeout_minutes` |
| `QUERY_EXECUTION` | `SUCCESS` / `FAILURE` | Evaluator executes question answering query in Tab 7. | `retrieval_mode`, `top_k`, `citations_count`, `evidence_count` |
| `DOCUMENT_INGESTION` | `SUCCESS` / `FAILURE` | User uploads and chunks documents in Tab 3. | `files_count`, `filenames`, `chunk_size`, `overlap` |
| `DOCUMENT_DELETION` | `SUCCESS` / `FAILURE` | Admin deletes a document from the corpus. | `doc_id`, `filename` |
| `BACKUP_CREATED` | `SUCCESS` / `FAILURE` | Backup snapshot archive created in Tab 9. | `backup_id`, `category`, `total_files`, `total_bytes` |
| `BACKUP_RESTORED` | `SUCCESS` / `FAILURE` | Admin executes disaster recovery restoration. | `backup_id`, `restored_files`, `snapshot_id` |
| `USER_CREATED` | `SUCCESS` | Admin authorizes a new user on the allowlist. | `new_user_id`, `email`, `role`, `notes` |
| `USER_UPDATED` | `SUCCESS` | User profile, notes, or active status updated. | `target_email`, `changes` |
| `USER_SUSPENDED` | `SUCCESS` | Admin changes account status to `SUSPENDED`. | `target_email`, `new_status` |
| `ROLE_CHANGED` | `SUCCESS` | Admin elevates or changes a user's role. | `target_email`, `old_role`, `new_role` |
| `PERMISSION_DENIED` | `DENIED` | Principal attempts operation without role permission. | `required_permission`, `attempted_action` |

---

## 4. Automated Zero-Credential-Leakage Redaction

The sanitization routine recursively traverses dictionaries and lists:
```python
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
```
If any key in `details` contains any of these substrings (case-insensitive), its value is overwritten with `"[REDACTED]"`.

---

## 5. Python API Reference

### 5.1 Writing an Event
```python
from graphrag_bench.user_audit import UserAuditLogger

audit = UserAuditLogger()
event = audit.log_event(
    user_id=user.user_id,
    email=user.email,
    role=user.role.value,
    action="QUERY_EXECUTION",
    status="SUCCESS",
    details={"mode": "hybrid", "citations": 3},
)
```

### 5.2 Querying Recent Events
```python
# Query recent 50 events for a specific user
events = audit.get_recent_events(limit=50, user_id="usr_abc123")

# Query recent denied access attempts
denied = audit.get_recent_events(limit=20, status="DENIED")
```

### 5.3 Aggregating Statistics
```python
stats = audit.get_summary_statistics()
# Returns:
# {
#   "total_events": 142,
#   "actions": {"AUTH_LOGIN": 24, "QUERY_EXECUTION": 110, ...},
#   "statuses": {"SUCCESS": 138, "DENIED": 4},
#   "users": {"jane@org.com": 85, "bob@org.com": 57}
# }
```

---

## 6. Log File Management & Rotation

- **Location:** `logs/user_audit.jsonl`
- **Permissions:** Recommended file permission `0600` (readable/writable only by the application service account).
- **Archival:** For long-term compliance, administrators should archive `logs/user_audit.jsonl` along with weekly backup snapshots.
