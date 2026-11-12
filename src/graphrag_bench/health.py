"""Health check subsystem for Graph RAG Benchmark.

Evaluates environment readiness across:
1. Python runtime version (3.11)
2. Virtual environment status
3. Critical package dependencies
4. Ollama server responsiveness
5. Configured LLM model availability
6. Configured embedding model availability
7. Required project directories
8. Configuration validity
"""

import importlib
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from graphrag_bench.config import Settings, get_settings
from graphrag_bench.logger import get_logger
from graphrag_bench.ollama_client import OllamaClient, OllamaError

logger = get_logger("health")


class HealthStatus(str, Enum):
    """Health check outcome status."""

    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


class HealthCheckResult(BaseModel):
    """Result container for an individual health check."""

    name: str = Field(description="Check identifier")
    status: HealthStatus = Field(description="Result status (PASS, WARNING, FAIL)")
    message: str = Field(description="Summary message")
    details: dict[str, Any] = Field(default_factory=dict, description="Diagnostic metadata")


class HealthReport(BaseModel):
    """Aggregated health report across all system checks."""

    overall_status: HealthStatus = Field(description="Overall health status")
    checks: list[HealthCheckResult] = Field(
        default_factory=list, description="List of check results"
    )
    summary: str = Field(description="Human readable summary statement")

    @property
    def is_ready(self) -> bool:
        """True if overall status is PASS or WARNING, False if FAIL."""
        return self.overall_status != HealthStatus.FAIL


class HealthChecker:
    """Orchestrates comprehensive environment and dependency validation."""

    def __init__(
        self,
        settings: Settings | None = None,
        ollama_client: OllamaClient | None = None,
        base_dir: Path | None = None,
    ) -> None:
        """Initialize health checker.

        Args:
            settings: Optional Settings instance. Defaults to get_settings().
            ollama_client: Optional OllamaClient instance.
            base_dir: Base directory of the repository. Defaults to current working dir.
        """
        self.settings = settings or get_settings()
        self.client = ollama_client or OllamaClient(settings=self.settings)
        self.base_dir = base_dir or Path.cwd()

    def check_python_version(self) -> HealthCheckResult:
        """Verify that Python version satisfies >= 3.11 requirement."""
        ver = sys.version_info
        version_str = f"{ver.major}.{ver.minor}.{ver.micro}"
        if ver.major == 3 and ver.minor == 11:
            return HealthCheckResult(
                name="Python Version",
                status=HealthStatus.PASS,
                message=f"Python version {version_str} satisfies target 3.11 exactly.",
                details={"version": version_str, "target": "3.11"},
            )
        elif ver.major == 3 and ver.minor >= 11:
            return HealthCheckResult(
                name="Python Version",
                status=HealthStatus.WARNING,
                message=f"Python version {version_str} is >= 3.11 (target is 3.11).",
                details={"version": version_str, "target": "3.11"},
            )
        else:
            return HealthCheckResult(
                name="Python Version",
                status=HealthStatus.FAIL,
                message=f"Python version {version_str} is incompatible (Python 3.11 required).",
                details={"version": version_str, "target": "3.11"},
            )

    def check_virtual_environment(self) -> HealthCheckResult:
        """Check whether the code is executing inside an active virtual environment."""
        is_venv = (
            (hasattr(sys, "base_prefix") and sys.prefix != sys.base_prefix)
            or (hasattr(sys, "real_prefix") and sys.real_prefix != sys.prefix)
            or "VIRTUAL_ENV" in os.environ
        )
        if is_venv:
            return HealthCheckResult(
                name="Virtual Environment",
                status=HealthStatus.PASS,
                message=f"Running in isolated virtual environment: {sys.prefix}",
                details={"prefix": sys.prefix, "in_venv": True},
            )
        return HealthCheckResult(
            name="Virtual Environment",
            status=HealthStatus.WARNING,
            message="Not running inside a virtual environment. Isolated .venv recommended.",
            details={"prefix": sys.prefix, "in_venv": False},
        )

    def check_dependencies(self) -> HealthCheckResult:
        """Verify that all core runtime and development dependencies are installed."""
        required = [
            ("requests", "requests"),
            ("dotenv", "python-dotenv"),
            ("pydantic", "pydantic"),
            ("pydantic_settings", "pydantic-settings"),
            ("streamlit", "streamlit"),
            ("pytest", "pytest"),
        ]
        missing = []
        installed_versions = {}

        for import_name, pkg_name in required:
            try:
                mod = importlib.import_module(import_name)
                ver = getattr(mod, "__version__", "unknown")
                installed_versions[pkg_name] = ver
            except ImportError:
                missing.append(pkg_name)

        if missing:
            return HealthCheckResult(
                name="Dependencies",
                status=HealthStatus.FAIL,
                message=f"Missing required packages: {', '.join(missing)}",
                details={"missing": missing, "installed": installed_versions},
            )
        return HealthCheckResult(
            name="Dependencies",
            status=HealthStatus.PASS,
            message=f"All {len(required)} core packages are installed.",
            details={"installed": installed_versions},
        )

    def check_configuration(self) -> HealthCheckResult:
        """Validate current environment settings."""
        try:
            safe_cfg = self.settings.to_safe_dict()
            return HealthCheckResult(
                name="Configuration",
                status=HealthStatus.PASS,
                message="Application settings loaded and validated successfully.",
                details=safe_cfg,
            )
        except Exception as exc:  # noqa: BLE001
            return HealthCheckResult(
                name="Configuration",
                status=HealthStatus.FAIL,
                message=f"Configuration validation failed: {exc}",
                details={"error": str(exc)},
            )

    def check_directories(self) -> HealthCheckResult:
        """Ensure required project directories exist."""
        required_dirs = ["logs", "data", "docs", "src", "tests", "scripts"]
        missing_dirs = []
        existing_dirs = []

        for d in required_dirs:
            path = self.base_dir / d
            if path.is_dir():
                existing_dirs.append(d)
            else:
                missing_dirs.append(d)

        if missing_dirs:
            return HealthCheckResult(
                name="Project Directories",
                status=HealthStatus.FAIL,
                message=f"Missing essential directories: {', '.join(missing_dirs)}",
                details={"missing": missing_dirs, "existing": existing_dirs},
            )
        return HealthCheckResult(
            name="Project Directories",
            status=HealthStatus.PASS,
            message="All project directories (logs, data, docs, src, tests, scripts) exist.",
            details={"existing": existing_dirs},
        )

    def check_ollama_server(self) -> HealthCheckResult:
        """Verify reachability of the Ollama server."""
        try:
            reachable = self.client.is_server_reachable()
            if reachable:
                return HealthCheckResult(
                    name="Ollama Server",
                    status=HealthStatus.PASS,
                    message=f"Ollama server is reachable at {self.client.base_url}.",
                    details={"base_url": self.client.base_url},
                )
            return HealthCheckResult(
                name="Ollama Server",
                status=HealthStatus.FAIL,
                message=f"Ollama server at {self.client.base_url} did not respond.",
                details={"base_url": self.client.base_url},
            )
        except Exception as exc:  # noqa: BLE001
            return HealthCheckResult(
                name="Ollama Server",
                status=HealthStatus.FAIL,
                message=f"Error checking Ollama server: {exc}",
                details={"error": str(exc)},
            )

    def check_llm_model(self) -> HealthCheckResult:
        """Verify that the configured LLM model is installed."""
        target_model = self.settings.ollama_llm_model
        try:
            if not self.client.is_server_reachable():
                return HealthCheckResult(
                    name="LLM Model",
                    status=HealthStatus.FAIL,
                    message=f"Cannot verify LLM model '{target_model}': Ollama server unreachable.",
                    details={"model": target_model},
                )

            installed = self.client.is_model_installed(target_model)
            if installed:
                return HealthCheckResult(
                    name="LLM Model",
                    status=HealthStatus.PASS,
                    message=f"Configured LLM model '{target_model}' is installed.",
                    details={"model": target_model},
                )
            return HealthCheckResult(
                name="LLM Model",
                status=HealthStatus.FAIL,
                message=f"Configured LLM model '{target_model}' is not installed in Ollama. Run: ollama pull {target_model}",
                details={"model": target_model},
            )
        except OllamaError as exc:
            return HealthCheckResult(
                name="LLM Model",
                status=HealthStatus.FAIL,
                message=f"Error checking LLM model '{target_model}': {exc}",
                details={"model": target_model, "error": str(exc)},
            )

    def check_embed_model(self) -> HealthCheckResult:
        """Verify that the configured embedding model is installed."""
        target_model = self.settings.ollama_embed_model
        try:
            if not self.client.is_server_reachable():
                return HealthCheckResult(
                    name="Embedding Model",
                    status=HealthStatus.FAIL,
                    message=f"Cannot verify embedding model '{target_model}': Ollama server unreachable.",
                    details={"model": target_model},
                )

            installed = self.client.is_model_installed(target_model)
            if installed:
                return HealthCheckResult(
                    name="Embedding Model",
                    status=HealthStatus.PASS,
                    message=f"Configured embedding model '{target_model}' is installed.",
                    details={"model": target_model},
                )
            return HealthCheckResult(
                name="Embedding Model",
                status=HealthStatus.FAIL,
                message=f"Configured embedding model '{target_model}' is not installed in Ollama. Run: ollama pull {target_model}",
                details={"model": target_model},
            )
        except OllamaError as exc:
            return HealthCheckResult(
                name="Embedding Model",
                status=HealthStatus.FAIL,
                message=f"Error checking embedding model '{target_model}': {exc}",
                details={"model": target_model, "error": str(exc)},
            )

    def run_all_checks(self) -> HealthReport:
        """Execute all health checks and return aggregated report."""
        checks = [
            self.check_python_version(),
            self.check_virtual_environment(),
            self.check_dependencies(),
            self.check_configuration(),
            self.check_directories(),
            self.check_ollama_server(),
            self.check_llm_model(),
            self.check_embed_model(),
        ]

        has_fail = any(c.status == HealthStatus.FAIL for c in checks)
        has_warning = any(c.status == HealthStatus.WARNING for c in checks)

        if has_fail:
            overall = HealthStatus.FAIL
            summary = (
                "Environment health check FAILED. Resolve blocking failures before proceeding."
            )
        elif has_warning:
            overall = HealthStatus.WARNING
            summary = "Environment health check PASSED with WARNINGS. Non-blocking alerts present."
        else:
            overall = HealthStatus.PASS
            summary = "All Phase 1 environment and integration health checks PASSED."

        return HealthReport(
            overall_status=overall,
            checks=checks,
            summary=summary,
        )
