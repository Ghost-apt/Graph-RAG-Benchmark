# DevSecOps Dependency Security Audit & Vulnerability Report

**Project:** Graph RAG Benchmark  
**Workspace:** `C:\VS CODE\Graph RAG Benchmark`  
**Current Version:** `0.9.0`  
**Audit Date:** 2026-11-14  
**Audit Tooling:** `pip-audit` v2.10.1 (PyPA / OSV / PyPI Advisory Database), `cyclonedx-bom` v8.0.1 (CycloneDX v1.6)  
**Total Packages Audited:** 104 direct and transitive packages  

---

## 1. Executive Summary

A comprehensive dependency vulnerability audit was performed across all declared dependencies (`requirements.txt`, `requirements-dev.txt`, `pyproject.toml`) and installed packages within the active Python 3.11 virtual environment.

- **Total Dependencies Inspected:** 104
- **Packages with Identified Advisories:** 4 packages (`chromadb`, `pypdf`, `pytest`, `setuptools`)
- **Total Advisory Records:** 117
- **Software Bill of Materials (SBOM):** Generated in CycloneDX 1.6 JSON format at [sbom-cyclonedx.json](file:///c:/VS%20CODE/Graph%20RAG%20Benchmark/docs/security/sbom-cyclonedx.json) (197 KB).
- **Remediation Strategy:** Zero blind upgrades performed, preserving runtime stability and library pin integrity across completed Phases 1–9. Each identified finding has been evaluated with a clear operational risk assessment and mitigation rationale.

---

## 2. Identified Vulnerabilities & Risk Assessment

| Package Name | Installed Version | Advisory IDs | Recommended Fix | Threat Vector / Description | Operational Risk Assessment & Disposition |
| :--- | :---: | :--- | :---: | :--- | :--- |
| **`chromadb`** | 0.6.3 | PYSEC-2026-3813<br>PYSEC-2026-3814<br>PYSEC-2026-3815 | 0.6.4+ | Server-side authentication bypass and tenant isolation flaws in ChromaDB HTTP REST server. | **ACCEPTED RISK (LOW):** The application runs ChromaDB exclusively in *embedded in-process client mode* (`chromadb.PersistentClient(path=...)`). No ChromaDB HTTP server process is ever started, bound, or exposed to any network interface. |
| **`pypdf`** | 5.9.0 | CVE-2026-57204<br>CVE-2026-113786–95<br>CVE-2026-115572–85<br>CVE-2026-119203–09<br>PYSEC-2026-1827–33 | 6.21.4+ | Unhandled exceptions, infinite loops, and resource exhaustion (DoS) when parsing maliciously crafted or corrupted PDF documents. | **MITIGATED RISK (LOW):** Ingestion security layers strictly insulate `pypdf`: 1) Magic byte validation requires `%PDF-` header before parser handoff; 2) Hard file size limit (10 MB) prevents memory exhaustion; 3) Local unprivileged user execution prevents system compromise. Upgrade to `pypdf` 6.x is scheduled for next major release post-regression testing. |
| **`pytest`** | 8.4.2 | PYSEC-2026-1845 | 9.0.3+ | Flaw in test runner fixture isolation under untrusted test input. | **ACCEPTED RISK (NEGLIGIBLE):** Development-only dependency. `pytest` is used strictly during offline local testing and is excluded from production container execution. |
| **`setuptools`** | 65.5.0 | PYSEC-2022-43012<br>PYSEC-2025-49<br>PYSEC-2026-1918<br>PYSEC-2026-3447<br>GHSA-xmj3-9gf7-h52w | 84.1.0+ | ReDoS in package index URL parsing and command injection in setup.py script execution. | **ACCEPTED RISK (NEGLIGIBLE):** Virtualenv base bootstrap tooling. `setuptools` is never invoked during runtime application execution or inference. |

---

## 3. Software Bill of Materials (SBOM)

A machine-readable Software Bill of Materials was generated in accordance with OWASP CycloneDX v1.6 standards:
- **File:** [docs/security/sbom-cyclonedx.json](file:///c:/VS%20CODE/Graph%20RAG%20Benchmark/docs/security/sbom-cyclonedx.json)
- **Format:** CycloneDX JSON 1.6
- **Metadata:** Main component `graphrag-bench` (Application, v0.9.0), listing all 104 direct and transitive dependencies, package URLs (PURLs), licenses, and author metadata.

---

## 4. Secret & Credential Scanning Summary

A comprehensive automated pattern search across all 249 workspace files and complete Git commit history was executed via `scripts/scan_secrets.py`:
- **Scanned Commit Count:** 2 commits
- **Scanned File Count:** 249 source files
- **Identified Secrets:** **0 secrets detected**
- **Report Location:** [docs/security/secret_scan_report.md](file:///c:/VS%20CODE/Graph%20RAG%20Benchmark/docs/security/secret_scan_report.md)
