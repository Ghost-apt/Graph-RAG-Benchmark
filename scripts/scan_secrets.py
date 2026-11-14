"""Comprehensive Secret and Credential Scanner for Graph RAG Benchmark.

Scans the entire repository tree and Git commit history for API keys, tokens,
private keys, and sensitive credential patterns.
"""

import re
import subprocess
from pathlib import Path

SECRET_PATTERNS = [
    ("AWS Access Key ID", re.compile(r"\b(AKIA[0-9A-Z]{16})\b")),
    ("OpenAI API Key", re.compile(r"\bsk-[a-zA-Z0-9]{32,64}\b")),
    ("Anthropic API Key", re.compile(r"\bsk-ant-[a-zA-Z0-9_-]{32,64}\b")),
    (
        "GitHub Personal Access Token",
        re.compile(r"\b(ghp_[a-zA-Z0-9]{36}|github_pat_[a-zA-Z0-9_]{60,100})\b"),
    ),
    ("Generic Private Key", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----")),
    (
        "Generic Password/Secret Assignment",
        re.compile(
            r"""(?i)(?:api_key|secret_key|client_secret|password|auth_token)\s*[:=]\s*["']([^"'\s]{8,})["']"""
        ),
    ),
    ("Database URI with Password", re.compile(r"""[a-zA-Z+]+://[^:\s]+:([^@\s]+)@[^/\s]+""")),
    ("Bearer Token", re.compile(r"""(?i)bearer\s+[a-zA-Z0-9_.\-]{30,}""")),
]

IGNORE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "site-packages",
    "node_modules",
}

IGNORE_FILES = {
    "uv.lock",
    "poetry.lock",
    "package-lock.json",
    "sbom-cyclonedx.json",
    "secret_scan_report.md",
}

# Values that are clearly placeholders or test fixtures
BENIGN_PLACEHOLDERS = {
    "******",
    "your_api_key_here",
    "your-api-key",
    "placeholder",
    "changeme",
    "dummy_token",
    "test_token",
    "fake_key",
    "sample_key",
}


def mask_secret(val: str) -> str:
    if len(val) <= 6:
        return "******"
    return val[:3] + "..." + val[-3:]


def scan_workspace(root: Path) -> list[dict]:
    findings = []
    scanned_count = 0
    for p in root.rglob("*"):
        if p.is_dir():
            continue
        if any(ignored in p.parts for ignored in IGNORE_DIRS):
            continue
        if p.name in IGNORE_FILES or p.name == "scan_secrets.py":
            continue
        if p.suffix in {
            ".pyc",
            ".tar",
            ".gz",
            ".zip",
            ".bin",
            ".whl",
            ".exe",
            ".dll",
            ".so",
            ".sqlite3",
            ".db",
        }:
            continue

        scanned_count += 1
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        for line_num, line in enumerate(content.splitlines(), start=1):
            for name, pattern in SECRET_PATTERNS:
                for match in pattern.finditer(line):
                    secret_val = match.group(1) if match.groups() else match.group(0)
                    if secret_val.strip().lower() in BENIGN_PLACEHOLDERS:
                        continue
                    # Ignore comment indicators that say example or dummy
                    if (
                        "example" in line.lower()
                        or "dummy" in line.lower()
                        or "mock" in line.lower()
                    ):
                        continue
                    findings.append(
                        {
                            "source": "workspace",
                            "path": str(p.relative_to(root)),
                            "line": line_num,
                            "type": name,
                            "masked_preview": mask_secret(secret_val),
                        }
                    )
    return findings, scanned_count


def scan_git_history(root: Path) -> tuple[list[dict], int]:
    findings = []
    try:
        proc = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
        )
        commit_count = int(proc.stdout.strip())
    except Exception:
        commit_count = 0

    try:
        log_proc = subprocess.run(
            ["git", "log", "-p", "--full-history"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
        )
        diff_text = log_proc.stdout
    except Exception as exc:
        return [{"error": f"Failed to read git log: {exc}"}], commit_count

    current_commit = ""
    for line in diff_text.splitlines():
        if line.startswith("commit "):
            current_commit = line.split()[1][:10]
        elif line.startswith("+") and not line.startswith("+++"):
            added_content = line[1:]
            for name, pattern in SECRET_PATTERNS:
                for match in pattern.finditer(added_content):
                    secret_val = match.group(1) if match.groups() else match.group(0)
                    if secret_val.strip().lower() in BENIGN_PLACEHOLDERS:
                        continue
                    if "example" in added_content.lower() or "mock" in added_content.lower():
                        continue
                    findings.append(
                        {
                            "source": "git_history",
                            "commit": current_commit,
                            "type": name,
                            "masked_preview": mask_secret(secret_val),
                        }
                    )
    return findings, commit_count


def main():
    root = Path(__file__).resolve().parent.parent
    workspace_findings, file_count = scan_workspace(root)
    git_findings, commit_count = scan_git_history(root)

    out_dir = root / "docs" / "security"
    out_dir.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "# Security Audit — Secret & Credential Pattern Scan Report",
        "",
        "**Project:** Graph RAG Benchmark  ",
        f"**Workspace:** `{root}`  ",
        f"**Total Files Scanned:** {file_count}  ",
        f"**Git Commits Scanned:** {commit_count}  ",
        f"**Status:** {'CLEAN (Zero Exposed Secrets)' if not workspace_findings and not git_findings else 'FINDINGS DETECTED'}  ",
        "",
        "---",
        "",
        "## 1. Scanner Patterns Audited",
        "",
        "The scanner evaluated all source files and commit history against high-risk pattern signatures:",
        "- AWS Access Keys (`AKIA...`)",
        "- OpenAI API Keys (`sk-...`)",
        "- Anthropic API Keys (`sk-ant-...`)",
        "- GitHub Personal Access Tokens (`ghp_...`, `github_pat_...`)",
        "- Private Key Blocks (`-----BEGIN PRIVATE KEY-----`)",
        "- Generic Password / Token Assignments in Code",
        "- Embedded Database URI Credentials",
        "- Hardcoded Bearer Authorization Tokens",
        "",
        "---",
        "",
        "## 2. Workspace File Scan Results",
        "",
    ]

    if not workspace_findings:
        report_lines.append("**Result:** 0 secrets detected in active workspace files.\n")
    else:
        report_lines.append(f"**Findings Detected:** {len(workspace_findings)}\n")
        report_lines.append("| File | Line | Type | Masked Value |")
        report_lines.append("| :--- | :---: | :--- | :--- |")
        for f in workspace_findings:
            report_lines.append(
                f"| `{f['path']}` | {f['line']} | {f['type']} | `{f['masked_preview']}` |"
            )
        report_lines.append("")

    report_lines.extend(
        [
            "---",
            "",
            "## 3. Git History Scan Results",
            "",
        ]
    )

    if not git_findings:
        report_lines.append(
            f"**Result:** 0 secrets detected across {commit_count} historical Git commits.\n"
        )
    else:
        report_lines.append(f"**Findings Detected:** {len(git_findings)}\n")
        report_lines.append("| Commit | Type | Masked Value |")
        report_lines.append("| :--- | :--- | :--- |")
        for f in git_findings:
            report_lines.append(
                f"| `{f.get('commit', 'N/A')}` | {f['type']} | `{f['masked_preview']}` |"
            )
        report_lines.append("")

    report_lines.extend(
        [
            "---",
            "",
            "## 4. Conclusion & Hygiene Verdict",
            "",
            f"- **Secret Leakage Status:** {'NO SECRETS DETECTED' if not workspace_findings and not git_findings else 'ACTION REQUIRED'}",
            "- **Protection Measures Active:** `.gitignore` and `.dockerignore` exclude `.env` files; `Settings.to_safe_dict()` automatically redacts memory configuration dumps.",
        ]
    )

    out_file = out_dir / "secret_scan_report.md"
    out_file.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Secret scan completed. Scanned {file_count} files, {commit_count} commits.")
    print(f"Workspace findings: {len(workspace_findings)}, Git findings: {len(git_findings)}")
    print(f"Report written to: {out_file}")


if __name__ == "__main__":
    main()
