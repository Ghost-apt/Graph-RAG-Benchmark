# Security Audit — Secret & Credential Pattern Scan Report

**Project:** Graph RAG Benchmark  
**Workspace:** `C:\VS CODE\Graph RAG Benchmark`  
**Total Files Scanned:** 249  
**Git Commits Scanned:** 2  
**Status:** CLEAN (Zero Exposed Secrets)  

---

## 1. Scanner Patterns Audited

The scanner evaluated all source files and commit history against high-risk pattern signatures:
- AWS Access Keys (`AKIA...`)
- OpenAI API Keys (`sk-...`)
- Anthropic API Keys (`sk-ant-...`)
- GitHub Personal Access Tokens (`ghp_...`, `github_pat_...`)
- Private Key Blocks (`-----BEGIN PRIVATE KEY-----`)
- Generic Password / Token Assignments in Code
- Embedded Database URI Credentials
- Hardcoded Bearer Authorization Tokens

---

## 2. Workspace File Scan Results

**Result:** 0 secrets detected in active workspace files.

---

## 3. Git History Scan Results

**Result:** 0 secrets detected across 2 historical Git commits.

---

## 4. Conclusion & Hygiene Verdict

- **Secret Leakage Status:** NO SECRETS DETECTED
- **Protection Measures Active:** `.gitignore` and `.dockerignore` exclude `.env` files; `Settings.to_safe_dict()` automatically redacts memory configuration dumps.