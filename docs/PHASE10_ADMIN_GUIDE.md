# Graph RAG Benchmark — Phase 10 Administrator Guide

**Release Version:** `v0.10.0` (Phase 10 — Real-User Pilot Deployment & Product Launch)  
**Target Audience:** System Administrators, DevSecOps Engineers, Platform Leads, SREs  
**Classification:** Operational Administration Guide  
**Status:** Approved & Enforced  

---

## 1. Overview & Administrator Responsibilities

As an Administrator of the Graph RAG Benchmark pilot deployment, you hold the highest privilege level (`Role.ADMIN`) with access to all 15 system permissions. Your core responsibilities include:
1. **Identity & Access Governance:** Managing the pilot user allowlist, approving evaluator access, assigning least-privilege roles, and suspending compromised or offboarded accounts.
2. **Security & Audit Oversight:** Monitoring attributable audit streams (`logs/user_audit.jsonl`), reviewing access denials, and ensuring zero credential leakage.
3. **Disaster Recovery:** Creating regular cryptographically verified backup archives, testing integrity, and executing authorized disaster restorations.
4. **GPU Resource Protection:** Managing concurrency slots to prevent out-of-memory (OOM) crashes on local NVIDIA GPU hardware (e.g. GeForce RTX 3050 with 4 GB VRAM).

---

## 2. Configuration & Environment Variables

The authentication and governance subsystem is configured via environment variables or a `.env` file in the project root:

| Environment Variable | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `GRAPHRAG_AUTH_MODE` | `str` | `none` | Active auth mode: `none` (local fallback), `dev` (persona switcher), `oidc` (production OIDC). |
| `GRAPHRAG_ADMIN_BOOTSTRAP_EMAIL` | `str` | `None` | Verified email address automatically assigned `ADMIN` role upon first OIDC sign-in. |
| `GRAPHRAG_USER_REGISTRY_PATH` | `str` | `data/users/users.json` | Path to persistent, atomic JSON user registry. |
| `GRAPHRAG_USER_AUDIT_LOG_PATH` | `str` | `logs/user_audit.jsonl` | Path to append-only attributable audit log stream. |
| `GRAPHRAG_SESSION_TIMEOUT_MINUTES` | `int` | `120` | Session inactivity timeout in minutes ($5$ to $1440$ min). |
| `GRAPHRAG_REQUIRE_EMAIL_VERIFIED` | `bool` | `True` | Rejects OIDC identities if `email_verified` claim is False. |
| `MAX_CONCURRENT_GENERATIONS` | `int` | `1` | Maximum parallel LLM generation operations allowed. |
| `MAX_CONCURRENT_BENCHMARKS` | `int` | `1` | Maximum parallel benchmark evaluations allowed. |

---

## 3. Initial Administrator Bootstrap Procedure

To eliminate fragile manual database editing or hardcoded initial credentials, the platform provides a verified claims bootstrap protocol:

```
[ Setup .env with ADMIN_BOOTSTRAP_EMAIL ] ---> [ Start Pilot Server in OIDC Mode ]
                                                         |
[ Admin Navigates to Dashboard ] <-----------------------+
       |
[ Clicks 'Sign in with Enterprise OIDC' ]
       |
[ IdP Authenticates Identity & Issues Token ]
       |
[ Backend Receives Claims: email="admin@organization.com" ]
       |
+-------------------------------------------------------------+
| Check: Does email match ADMIN_BOOTSTRAP_EMAIL?              |
+-------------------------------------------------------------+
       |
     [ YES ]
       |
[ System Automatically Provisions Account as Role.ADMIN ]
[ Logs USER_CREATED with note 'Initial Administrator Bootstrap' ]
[ Redirects Admin to Full Dashboard with Full Privileges ]
```

### 3.1 Step-by-Step Bootstrap Instructions
1. Configure your administrator email in `.env`:
   ```bash
   GRAPHRAG_AUTH_MODE=oidc
   GRAPHRAG_ADMIN_BOOTSTRAP_EMAIL=lead.evaluator@organization.com
   ```
2. Verify OIDC secrets in `.streamlit/secrets.toml`:
   ```bash
   python scripts/check_auth_config.py
   ```
3. Launch the pilot server:
   ```powershell
   .\scripts\start_pilot.ps1 -AuthMode oidc
   ```
4. Navigate to `http://localhost:8501` and sign in using `lead.evaluator@organization.com`.
5. Verify that your sidebar user badge displays **ADMIN** in red. All subsequent uninvited logins will be denied.

---

## 4. User Allowlist Management (Tab 9 Section H)

Administrators manage user accounts directly within the interactive dashboard under **Tab 9 (Production Operations) -> Section H (User Management & Identity Governance)**:

### 4.1 Pre-Registering New Pilot Evaluators
1. In Tab 9 Section H, click the **"➕ Authorize New User"** tab.
2. Enter the evaluator's enterprise email address (e.g. `jane.doe@organization.com`).
3. Enter their full name and team/notes (e.g. `NLP Evaluation Team`).
4. Select their initial role:
   - **`researcher` (Recommended):** Grants query execution, document ingestion, and benchmark evaluation.
   - **`viewer`:** Grants read-only access for observers.
5. Click **"Authorize & Add User"**.
6. The user is immediately appended to `data/users/users.json` and a `USER_CREATED` audit event is logged. When the user logs in via OIDC, they are instantly recognized and granted access.

### 4.2 Updating Roles or Suspending Accounts
1. In Tab 9 Section H, select **"📋 User Directory & Access Control"**.
2. Select the target user from the **"Select Target User"** dropdown.
3. To change their role: select the new role (`viewer`, `researcher`, or `admin`) and click **"Update Role"**.
4. To suspend an account: select **`suspended`** from the Account Status dropdown and click **"Update Status"**.
   - Suspended accounts are immediately blocked on their next action or login attempt.
   - To reactivate, change status back to **`active`**.

---

## 5. Reviewing Attributable Audit Streams

Every security-sensitive action across the platform is logged in real-time to `logs/user_audit.jsonl`.

### 5.1 Interactive Audit Explorer (Tab 9 Section H3)
The dashboard provides a dedicated audit viewer with:
- **Action Filters:** Filter by `AUTH_LOGIN`, `AUTH_LOGOUT`, `AUTH_FAILURE`, `QUERY_EXECUTION`, `DOCUMENT_INGESTION`, `DOCUMENT_DELETION`, `BACKUP_CREATED`, `BACKUP_RESTORED`, `USER_CREATED`, `ROLE_CHANGED`, or `PERMISSION_DENIED`.
- **Status Filters:** Filter by `SUCCESS`, `FAILURE`, or `DENIED`.
- **Summary Metrics:** Total events recorded, successful logins, queries executed, and denied operations.
- **Payload Inspection:** View JSON metadata associated with each event.

### 5.2 Command-Line Audit Inspection
You can also inspect audit streams using standard Python CLI tools:
```powershell
# View summary statistics
python -c "from graphrag_bench.user_audit import UserAuditLogger; print(UserAuditLogger().get_summary_statistics())"

# Query the last 5 denied operations
python -c "from graphrag_bench.user_audit import UserAuditLogger; print([e.model_dump() for e in UserAuditLogger().get_recent_events(limit=5, status='DENIED')])"
```

---

## 6. Disaster Recovery & Backup Operations

The platform includes a zero-external-dependency disaster recovery engine:

### 6.1 Creating Backups
- **Via Dashboard:** Navigate to Tab 9 Section E, select category (`FULL`, `CORPUS_ONLY`, `DATABASE_ONLY`), enter an optional note, and click **"💾 Create Backup Snapshot"**.
- **Via CLI:**
  ```powershell
  python scripts/backup.py --category full --description "Weekly pilot snapshot"
  ```
- Backups are stored as `.tar.gz` archives in `data/backups/` containing an immutable `manifest.json` with SHA-256 hashes for every archived file.

### 6.2 Restoring from Backup (Admin Only)
Restoring operations can overwrite active databases. The restore procedure requires:
1. **Integrity Verification:** Click **"🔍 Verify Integrity"** to validate SHA-256 checksums and TarSlip path traversal safety before extracting.
2. **Dry-Run Preview:** Click **"👁️ Dry-Run Preview"** to inspect target file paths and byte sizes.
3. **Explicit Confirmation Checkbox:** Check the safety acknowledgement checkbox.
4. **Execution:** Click **"⚡ Restore From Backup"**.
   - The system automatically creates a pre-restore safety snapshot before extracting files, guaranteeing non-destructive recovery.

---

## 7. GPU & Concurrency Management

When running on local hardware (e.g. NVIDIA RTX 3050 with 4 GB VRAM):
- Dual concurrent inference consumes ~3.2 GB of VRAM.
- To prevent CUDA out-of-memory errors during high-load pilot testing:
  - Keep `MAX_CONCURRENT_GENERATIONS=1` in production settings.
  - Additional concurrent requests enter an in-memory wait queue protected by `ConcurrencyLimiter` and acquire slots as previous queries complete.
