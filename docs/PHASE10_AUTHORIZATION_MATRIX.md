# Graph RAG Benchmark — Phase 10 Authorization Matrix

**Document Version:** `1.0.0`  
**System Target Release:** `v0.10.0` (Phase 10 — Real-User Pilot Deployment & Product Launch)  
**Classification:** Security Architecture & Access Control Specification  
**Status:** Approved & Enforced  

---

## 1. Overview & Security Philosophy

The Graph RAG Benchmark platform employs strict **Role-Based Access Control (RBAC)** adhering to the **Principle of Least Privilege (PoLP)**. Every user identity authenticated via OpenID Connect or the local development persona provider is mapped to exactly one canonical role.

Permissions are granular, functional, and evaluated both at the **UI presentation layer** (disabling or hiding sensitive controls) and at the **server-side execution layer** (raising `PermissionDeniedError` if an unauthorized action is attempted).

---

## 2. Canonical Roles

The platform defines three canonical roles designed specifically for empirical AI evaluation and research operations:

| Role Identifier | Role Name | Intended Audience | Core Purpose |
| :--- | :--- | :--- | :--- |
| `admin` | **System Administrator** | DevOps, SRE, Lead Evaluators | Full platform governance, user onboarding, role elevation, disaster recovery restore, system configuration. |
| `researcher` | **Research Scientist / Evaluator** | AI/ML Engineers, Evaluators, Domain Analysts | Ingestion of benchmark documents, retrieval experimentation, grounded QA execution, 60-query benchmark evaluation, backup snapshot creation. |
| `viewer` | **Guest / Read-Only Stakeholder** | Project Managers, Auditors, Observers | Exploratory query execution, viewing knowledge graphs, inspecting benchmark metrics and distribution charts. Cannot modify data or settings. |

---

## 3. Granular Functional Permissions

The system defines **15 granular permissions** organized into five operational domains:

| # | Permission Identifier | Domain | Description |
| :---: | :--- | :--- | :--- |
| 1 | `documents.read` | Document Management | View ingested documents, chunk counts, metadata, and token statistics. |
| 2 | `documents.upload` | Document Management | Upload new documents (PDF, DOCX, TXT, MD) and trigger sliding-window chunking. |
| 3 | `documents.delete` | Document Management | Delete documents and associated chunks from the persistent repository. |
| 4 | `retrieval.execute` | Retrieval & QA | Execute semantic search, graph traversal, and hybrid RRF retrieval. |
| 5 | `answers.generate` | Retrieval & QA | Generate grounded answers using local LLM inference with citation validation. |
| 6 | `answers.export` | Retrieval & QA | Export grounded Q&A results, citations, and evidence payloads to JSON/Markdown. |
| 7 | `benchmarks.read` | Comparative Evaluation | View benchmark datasets, splits, historical evaluation runs, and comparison charts. |
| 8 | `benchmarks.execute` | Comparative Evaluation | Execute comparative evaluation runs across Vector, Graph, and Hybrid RAG modalities. |
| 9 | `benchmarks.export` | Comparative Evaluation | Export benchmark metrics, paired Wilcoxon p-values, and confidence intervals to CSV/JSON. |
| 10 | `operations.read` | Operations & SRE | View production health state, telemetry dashboards, latency percentiles, and backup lists. |
| 11 | `operations.backup` | Operations & SRE | Create cryptographically verified `.tar.gz` backup snapshots with SHA-256 manifests. |
| 12 | `operations.restore` | Operations & SRE | Execute disaster recovery restoration from verified backup archives into active storage. |
| 13 | `users.read` | Identity & Governance | View registered user accounts, active sessions, and attributable audit logs. |
| 14 | `users.manage` | Identity & Governance | Authorize new users, modify assigned roles, suspend/reactivate accounts, revoke access. |
| 15 | `settings.manage` | System Configuration | Modify runtime configuration, model selection, threshold calibration, and resource bounds. |

---

## 4. Role-to-Permission Mapping Matrix

The following table details the authorized permissions for each canonical role. The matrix is enforced in code via `ROLE_PERMISSIONS` in `src/graphrag_bench/access_control.py`:

| Permission Identifier | Admin (`admin`) | Researcher (`researcher`) | Viewer (`viewer`) |
| :--- | :---: | :---: | :---: |
| `documents.read` | ✅ **Granted** | ✅ **Granted** | ✅ **Granted** |
| `documents.upload` | ✅ **Granted** | ✅ **Granted** | ❌ *Denied* |
| `documents.delete` | ✅ **Granted** | ❌ *Denied* | ❌ *Denied* |
| `retrieval.execute` | ✅ **Granted** | ✅ **Granted** | ✅ **Granted** |
| `answers.generate` | ✅ **Granted** | ✅ **Granted** | ✅ **Granted** |
| `answers.export` | ✅ **Granted** | ✅ **Granted** | ✅ **Granted** |
| `benchmarks.read` | ✅ **Granted** | ✅ **Granted** | ✅ **Granted** |
| `benchmarks.execute` | ✅ **Granted** | ✅ **Granted** | ❌ *Denied* |
| `benchmarks.export` | ✅ **Granted** | ✅ **Granted** | ✅ **Granted** |
| `operations.read` | ✅ **Granted** | ✅ **Granted** | ❌ *Denied* |
| `operations.backup` | ✅ **Granted** | ✅ **Granted** | ❌ *Denied* |
| `operations.restore` | ✅ **Granted** | ❌ *Denied* | ❌ *Denied* |
| `users.read` | ✅ **Granted** | ❌ *Denied* | ❌ *Denied* |
| `users.manage` | ✅ **Granted** | ❌ *Denied* | ❌ *Denied* |
| `settings.manage` | ✅ **Granted** | ❌ *Denied* | ❌ *Denied* |
| **Total Permissions Granted** | **15 / 15** | **9 / 15** | **5 / 15** |

### 4.1 Key Security Distinctions
- **Disaster Recovery Isolation:** Only `ADMIN` can invoke `operations.restore`. A compromised researcher or viewer account cannot overwrite canonical datasets or inject historic snapshots.
- **Document Protection:** Only `ADMIN` can execute `documents.delete`. Researchers can contribute documents (`documents.upload`), but cannot remove existing corpus documents without administrative confirmation.
- **Identity Governance:** Only `ADMIN` can access Tab 9 Section H to authorize new users or modify RBAC assignments.

---

## 5. Tab-by-Tab Access Governance

| Tab Index & Name | Viewer (`viewer`) | Researcher (`researcher`) | Admin (`admin`) |
| :--- | :--- | :--- | :--- |
| **Tab 1: Knowledge Graph Explorer** | Full interactive visualization, path inspection | Full interactive visualization, path inspection | Full visualization & graph re-indexing |
| **Tab 2: Embeddings & Vector Search** | Query semantic search, view rankings | Query semantic search, view rankings | Query search, reset ChromaDB collection |
| **Tab 3: Document Ingestion** | Read document library; upload/delete disabled | Upload documents, configure chunking | Upload, configure chunking, delete documents |
| **Tab 4: Health & Diagnostics** | View component pass/fail health checks | View health checks & detailed diagnostic outputs | Full diagnostics & health failure injection |
| **Tab 5: Graph Retrieval Lab** | Execute multi-hop traversal queries | Execute traversal queries & path rank tuning | Full traversal execution & tuning |
| **Tab 6: Hybrid Retrieval Lab** | Test reciprocal rank fusion queries | Run hybrid queries & tune weight distributions | Full hybrid execution & tuning |
| **Tab 7: Grounded RAG Chat** | Formulate questions, inspect citations | Formulate questions, inspect citations & claims | Full question answering & guardrail overrides |
| **Tab 8: Comparative Benchmark** | View aggregate tables & Wilcoxon charts | Run benchmark splits & export metrics | Run benchmark splits & manage corpus splits |
| **Tab 9: Production Operations** | Tab visible with read-only notice | Create backup snapshot, inspect telemetry | Full operations, execute restore, manage users |

---

## 6. Server-Side Enforcement Mechanisms

### 6.1 Programmatic Check Function
```python
from graphrag_bench.access_control import Permission, check_permission

# Explicit check raising PermissionDeniedError if unauthorized
check_permission(role=user.role, permission=Permission.OPERATIONS_RESTORE, user_id=user.user_id)
```

### 6.2 Declarative Function Decorator
```python
from graphrag_bench.access_control import Permission, require_permission


@require_permission(Permission.USERS_MANAGE)
def update_user_role(target_user_id: str, new_role: Role, role: Role = Role.ADMIN) -> UserAccount:
    # Function body executes ONLY if caller possesses USERS_MANAGE permission
    ...
```

### 6.3 Attributable Violation Logging
Whenever a permission check fails, the incident is immediately recorded in `logs/user_audit.jsonl` with status `DENIED`:
```json
{
  "event_id": "c8b49e1a-5f32-4d7a-8b19-e932b1234567",
  "timestamp": "2026-11-14T10:15:30.123456Z",
  "user_id": "usr_7a18f9c2d1b045e3",
  "email": "viewer@organization.com",
  "role": "viewer",
  "action": "DOCUMENTS_UPLOAD",
  "status": "DENIED",
  "details": {"reason": "insufficient_role_permission", "required_permission": "documents.upload"},
  "ip_address": "192.168.1.105"
}
```

---

## 7. Account Lifecycle Governance

1. **Bootstrap Admin:** The first account matching `ADMIN_BOOTSTRAP_EMAIL` is automatically provisioned as `ADMIN`.
2. **Allowlist Requirement:** Unregistered users attempting OIDC login are stopped with an informative notice.
3. **Suspension:** Administrators can toggle account status from `ACTIVE` to `SUSPENDED` instantly revoking access without deleting historical attribution.
