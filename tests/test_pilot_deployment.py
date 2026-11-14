"""Integration and validation tests for Phase 10 pilot deployment artifacts.

Tests:
- Secrets template syntax and parameters
- Git and Docker secrets protection
- Auth config checking utility
- Pilot health check execution
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from unittest.mock import patch

from scripts.check_auth_config import check_auth_configuration
from scripts.pilot_health_check import run_pilot_health_check


class TestPilotDeployment:
    """Deployment artifact tests."""

    def test_secrets_template_validity(self) -> None:
        """Verify .streamlit/secrets.toml.example exists, parses, and defines required fields."""
        template_path = Path(".streamlit/secrets.toml.example")
        assert template_path.exists(), ".streamlit/secrets.toml.example must exist"

        with open(template_path, "rb") as f:
            data = tomllib.load(f)

        assert "auth" in data, "Secrets template must contain [auth] table"
        auth = data["auth"]
        assert "client_id" in auth
        assert "client_secret" in auth
        assert "server_metadata_url" in auth
        assert "redirect_uri" in auth

    def test_secrets_protection_in_ignore_files(self) -> None:
        """Verify .streamlit/secrets.toml is protected in .gitignore and .dockerignore."""
        gitignore = Path(".gitignore").read_text(encoding="utf-8")
        assert ".streamlit/secrets.toml" in gitignore

        dockerignore = Path(".dockerignore").read_text(encoding="utf-8")
        assert ".streamlit/secrets.toml" in dockerignore

    def test_check_auth_config_dev_mode(self) -> None:
        """Verify check_auth_configuration passes when AUTH_MODE=dev."""
        with patch.dict("os.environ", {"GRAPHRAG_AUTH_MODE": "dev"}):
            code = check_auth_configuration()
            assert code == 0

    def test_check_auth_config_none_mode(self) -> None:
        """Verify check_auth_configuration passes when AUTH_MODE=none."""
        with patch.dict("os.environ", {"GRAPHRAG_AUTH_MODE": "none"}):
            code = check_auth_configuration()
            assert code == 0

    def test_pilot_health_check_runs(self) -> None:
        """Verify pilot_health_check executes and evaluates system status."""
        code = run_pilot_health_check(as_json=True)
        # Should return 0 (healthy) or 1 (warning if model not running), without crashing
        assert code in (0, 1)
