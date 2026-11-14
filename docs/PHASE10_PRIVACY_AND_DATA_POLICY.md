# Graph RAG Benchmark — Phase 10 Privacy & Data Governance Policy

**Document Version:** `1.0.0`  
**Release Target:** `v0.10.0` (Phase 10 — Real-User Pilot Deployment & Product Launch)  
**Classification:** Information Security & Data Governance Policy  
**Status:** Approved & Enforced  

---

## 1. Purpose & Scope

The **Graph RAG Benchmark** platform is designed for empirical research and comparative evaluation of information retrieval and language generation models. In a multi-user pilot environment, maintaining strict data integrity, user privacy, and operational security is paramount.

This policy defines:
1. The architectural separation between **individual session privacy** and the **shared research corpus**.
2. Categories of data collected, processed, and stored by the platform.
3. The zero-credential-leakage guarantee.
4. Data retention, export, and deletion protocols.

---

## 2. Multi-User Session Isolation vs Shared Research Workspace

The platform operates as a **collaborative research platform** with explicit boundary governance:

```
+-----------------------------------------------------------------------------------+
|                           APPLICATION PRIVACY BOUNDARIES                          |
+-----------------------------------------------------------------------------------+

     [ User A Session ]                                     [ User B Session ]
  - Private Streamlit State                               - Private Streamlit State
  - Unsubmitted Draft Questions                           - Unsubmitted Draft Questions
  - Local Slider Settings                                 - Local Slider Settings
  - Interactive Filter Selections                         - Interactive Filter Selections
            |                                                      |
            +---------------------------+--------------------------+
                                        |
                                        v
                 +-----------------------------------------------+
                 |        SHARED RESEARCH WORKSPACE (RBAC)       |
                 +-----------------------------------------------+
                 |  • Canonical Documents (`data/processed/`)    |
                 |  • Vector DB Collection (`data/chroma/`)      |
                 |  • Knowledge Graph (`data/knowledge_graph/`)  |
                 |  • Benchmark Datasets (`data/benchmark/`)     |
                 |  • Audit Log Stream (`logs/user_audit.jsonl`) |
                 +-----------------------------------------------+
```

### 2.1 Private Session State
- Each user connection maintains an isolated Python execution thread and `st.session_state` dictionary.
- Draft inputs, unsubmitted queries, and transient UI selections are private to that browser tab and never accessible to other concurrent users.

### 2.2 Shared Research Assets
- Ingested documents, vector embeddings, and knowledge graph entities are shared across all authorized researchers to ensure experimental consistency and avoid redundant processing.
- Changes to shared assets (uploading documents, building graphs, or restoring backups) require explicit RBAC permissions and are recorded in the central audit stream.

---

## 3. Data Categories Collected & Processed

| Category | Specific Attributes Stored | Purpose | Storage Location |
| :--- | :--- | :--- | :--- |
| **User Identity** | `user_id` (deterministic opaque hash), `email`, `name`, `role`, `status`, `created_at`, `last_login_at`. | Access control, authentication, audit attribution. | `data/users/users.json` |
| **Evaluation Queries** | Question text, selected retrieval mode (`vector`, `graph`, `hybrid`), execution latency, token counts. | Benchmark reproducibility, quality evaluation. | `logs/user_audit.jsonl`, `data/phase8_benchmark/` |
| **Document Content** | File text, word counts, chunk boundaries, metadata. | Information retrieval, embedding indexing, entity extraction. | `data/uploads/`, `data/processed/` |
| **Cryptographic Indexes** | 768-dim dense embeddings (`nomic-embed-text`), entity nodes, relationship triples, confidence scores. | Fast similarity search and multi-hop graph reasoning. | `data/chroma/`, `data/knowledge_graph/` |
| **Security Audit Logs** | Event UUID, UTC timestamp, user ID, email, role, action, outcome (`SUCCESS`/`DENIED`), IP address. | Compliance, forensics, security incident tracking. | `logs/user_audit.jsonl` |

---

## 4. Zero-Credential-Leakage Guarantee

The platform enforces technical safeguards to prevent sensitive credentials from ever being written to logs, disk, or exported summaries:

1. **Excluded Secrets in VCS:** `.streamlit/secrets.toml` is permanently blocked in `.gitignore` and `.dockerignore`.
2. **Automated Log Scrubbing:** Every payload dictionary passed to `UserAuditLogger` is recursively filtered by `sanitize_audit_details()`. Any key matching:
   - `password`
   - `token`, `access_token`, `id_token`, `refresh_token`
   - `secret`, `client_secret`, `cookie_secret`
   - `api_key`
   - `credential`
   - `authorization`, `bearer`
   is automatically replaced with `[REDACTED]`.
3. **Redacted Configuration Display:** Tab 9 Section F displays configuration parameters via `Settings.to_safe_dict()`, ensuring client secrets and tokens are masked.

---

## 5. Data Retention & Deletion Protocols

### 5.1 Document Library Management
- Administrators can delete individual documents via Tab 3 Section F. Deleting a document removes its processed JSON metadata and associated chunks from the local repository.
- Full index resets can be executed in Tab 2 (ChromaDB collection purge) or Tab 9 (Disaster Recovery restoration).

### 5.2 User Account Offboarding
- If a pilot evaluator leaves the organization or project, an Administrator updates their status to `SUSPENDED` in Tab 9 Section H.
- Suspended users are immediately denied access on all subsequent requests.
- Historical audit events remain attributable to the user's historical ID to maintain scientific integrity and audit trail completeness.

### 5.3 Audit Log Retention
- `logs/user_audit.jsonl` is retained for the duration of the pilot evaluation project.
- Logs can be exported to JSON/CSV by Administrators for archival or scientific review.

---

## 6. Compliance & Researcher Agreement
By participating in the Graph RAG Benchmark pilot, evaluators agree that:
1. Ingested documents must comply with organizational copyright and data classification policies.
2. Formulated queries are logged for evaluation attribution and benchmark validation.
3. Exploitation attempts, prompt injection attacks, or unauthorized permission elevation are logged and flagged for security review.
