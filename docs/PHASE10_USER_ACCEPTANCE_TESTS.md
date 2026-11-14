# Graph RAG Benchmark — Phase 10 User Acceptance Testing (UAT)

**Document Version:** `1.0.0`  
**Release Target:** `v0.10.0` (Phase 10 — Real-User Pilot Deployment & Product Launch)  
**Status:** Approved for Formal Execution  

---

## 1. Executive Summary & Objective

This document outlines the formal **User Acceptance Testing (UAT)** plan for Graph RAG Benchmark Phase 10. The objective is to validate that the platform satisfies all functional, security, performance, and user-experience criteria across the three canonical roles (`ADMIN`, `RESEARCHER`, `VIEWER`).

---

## 2. Test Environment Specifications

- **Workspace:** `C:\VS CODE\Graph RAG Benchmark`
- **Application Target Version:** `v0.10.0`
- **Host Hardware:** Local Workstation with NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)
- **Local AI Models:** `qwen3:8b` (LLM & Graph Extractor), `nomic-embed-text` (Embeddings)
- **Primary Browser:** Google Chrome v110+ / Microsoft Edge v110+

---

## 3. UAT Test Cases Matrix

### UAT-01: Unauthenticated Visitor Redirection & Welcome Page
- **Role:** Unauthenticated Visitor
- **Pre-conditions:** Browser cache cleared, no session cookies.
- **Action:** Navigate to `http://localhost:8501`.
- **Expected Outcome:**
  1. User is presented with the Welcome & Secure Sign-In page.
  2. None of the 9 navigation tabs or underlying data are visible.
  3. "Sign in with Enterprise OIDC" or dev persona selector is displayed.
- **Status:** **PASS**

---

### UAT-02: Initial Administrator Bootstrap via Verified Claims
- **Role:** First Administrator
- **Pre-conditions:** `GRAPHRAG_ADMIN_BOOTSTRAP_EMAIL=pilot.admin@corp.org` in `.env`. Empty registry or first sign-in.
- **Action:** Sign in using `pilot.admin@corp.org`.
- **Expected Outcome:**
  1. System detects matching bootstrap email.
  2. Account is provisioned with `Role.ADMIN` and `UserStatus.ACTIVE`.
  3. Sidebar displays **SIGNED IN AS: Pilot Admin** with red **ADMIN** role badge.
  4. Tab 9 Section H is accessible with full user management tools.
  5. Audit log records `USER_CREATED` with detail `"Initial administrator bootstrap"`.
- **Status:** **PASS**

---

### UAT-03: Uninvited User Rejection & Allowlist Enforcement
- **Role:** Uninvited Visitor (`intruder@outside.com`)
- **Pre-conditions:** User is not pre-registered in `data/users/users.json`.
- **Action:** Attempt sign-in with `intruder@outside.com`.
- **Expected Outcome:**
  1. Authentication is blocked.
  2. Screen displays: *"🚫 Access Restricted: Account is not on the invited pilot allowlist."*
  3. No application tabs or data are exposed.
  4. Audit log records `AUTH_FAILURE` with status `DENIED`.
- **Status:** **PASS**

---

### UAT-04: Researcher Role Ingestion & Grounded QA with Citations
- **Role:** `RESEARCHER`
- **Action:**
  1. Navigate to Tab 3 (Document Ingestion). Upload a `.txt` file and click **"Process & Ingest"**.
  2. Navigate to Tab 7 (Grounded RAG Chat). Select `Hybrid (RRF)` mode and ask *"What is the main advantage of Graph RAG over Vector RAG?"*.
- **Expected Outcome:**
  1. Ingestion succeeds; chunks are stored and embeddings indexed.
  2. Grounded answer is generated with inline citation footnotes `[1]`, `[2]`.
  3. Citation Footnotes section lists supporting chunk texts and similarity scores.
  4. In Tab 3 Section F, the Delete Document button is disabled (Researcher cannot delete).
  5. Audit log records `DOCUMENT_INGESTION` and `QUERY_EXECUTION`.
- **Status:** **PASS**

---

### UAT-05: Viewer Role Read-Only Restriction
- **Role:** `VIEWER`
- **Action:**
  1. Navigate to Tab 3 (Document Ingestion).
  2. Navigate to Tab 8 (Comparative Benchmark).
  3. Navigate to Tab 9 (Operations).
- **Expected Outcome:**
  1. In Tab 3, the "Process & Ingest Documents" button is disabled with notice: *"🔒 Document ingestion is restricted to RESEARCHER and ADMIN roles."*
  2. In Tab 8, benchmark execution buttons are disabled (read-only view of historical metrics).
  3. In Tab 9, disaster recovery restore is disabled and User Management displays access restriction notice.
- **Status:** **PASS**

---

### UAT-06: Evaluator Running Comparative Benchmark Lab
- **Role:** `RESEARCHER` or `ADMIN`
- **Action:** Navigate to Tab 8. Select **Validation Split (15 queries)**. Click **"🚀 Run Comparative Benchmark"**.
- **Expected Outcome:**
  1. Benchmark executes across Vector, Graph, and Hybrid RAG.
  2. Displays comparative metric table (MRR, P@5, R@5, NDCG@5).
  3. Computes paired Wilcoxon signed-rank test and 95% bootstrap confidence intervals.
  4. Export CSV button downloads the run results.
- **Status:** **PASS**

---

### UAT-07: Inactivity Session Expiration & Re-Authentication
- **Role:** Authenticated Evaluator
- **Pre-conditions:** Session inactivity exceeds `SESSION_TIMEOUT_MINUTES`.
- **Action:** User interacts with the dashboard after timeout period.
- **Expected Outcome:**
  1. System detects elapsed time > timeout window.
  2. Session state is cleared and `st.logout()` is called.
  3. User is returned to Welcome page with warning: *"⏱️ Your session has expired due to inactivity."*
  4. Audit log records `AUTH_SESSION_EXPIRED`.
- **Status:** **PASS**

---

### UAT-08: Administrator Account Suspension & Real-Time Denial
- **Role:** `ADMIN` managing Evaluator `bob@corp.org`
- **Action:**
  1. In Tab 9 Section H, Admin changes `bob@corp.org` status to `suspended`.
  2. Bob attempts to submit a query or refresh.
- **Expected Outcome:**
  1. Admin sees success confirmation.
  2. Bob is blocked immediately with message: *"🚫 Account Suspended"*.
  3. Audit log records `USER_SUSPENDED` and subsequent `AUTH_FAILURE`.
- **Status:** **PASS**

---

### UAT-09: Disaster Recovery Backup Verification, Preview, & Restoration
- **Role:** `ADMIN`
- **Action:**
  1. In Tab 9 Section E, click **"💾 Create Backup Snapshot"**.
  2. In Section F, select the newly created backup.
  3. Click **"🔍 Verify Integrity"**, then **"👁️ Dry-Run Preview"**.
  4. Check the confirmation checkbox and click **"⚡ Restore From Backup"**.
- **Expected Outcome:**
  1. Archive `.tar.gz` created with SHA-256 manifest.
  2. Integrity check reports all hashes match and no path traversal vulnerabilities.
  3. Preview lists exact file counts and byte sizes.
  4. Restore creates an automatic pre-restore safety snapshot before safely restoring files.
  5. Audit log records `BACKUP_CREATED` and `BACKUP_RESTORED`.
- **Status:** **PASS**

---

### UAT-10: Attributable Audit Log Verification & Credential Redaction
- **Role:** `ADMIN`
- **Action:** Navigate to Tab 9 Section H3 (Audit Stream). Search for recent actions.
- **Expected Outcome:**
  1. Table displays timestamp, user email, role, action, and status.
  2. Summary metrics count total events, logins, and queries.
  3. All metadata dictionaries show zero leaked tokens, passwords, or secret strings (`[REDACTED]`).
- **Status:** **PASS**

---

## 4. Formal Acceptance Sign-Off

| Milestone | Target Gate | Result | Auditor Sign-Off |
| :--- | :---: | :---: | :---: |
| UAT-01 through UAT-10 | 100% Pass Rate | **PASSED (10/10)** | ✅ Senior Security & QA Lead |
| Full Pytest Suite | 417 Passed Tests | **PASSED (417/417)** | ✅ SRE / QA Lead |
| Zero Security Vulnerabilities | 0 High / 0 Critical | **PASSED** | ✅ DevSecOps Specialist |
