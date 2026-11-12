"""Unit tests for the Health Check system (graphrag_bench.health).

Uses mocks for OllamaClient and filesystem to thoroughly test PASS, WARNING,
and FAIL states for all health check rules.
"""

import sys

import pytest

from graphrag_bench.config import Settings
from graphrag_bench.health import HealthChecker, HealthStatus
from graphrag_bench.ollama_client import OllamaClient


@pytest.fixture
def mock_client(mocker):
    """Mocked OllamaClient instance."""
    client = mocker.create_autospec(OllamaClient, instance=True)
    client.base_url = "http://localhost:11434"
    client.is_server_reachable.return_value = True
    client.is_model_installed.return_value = True
    client.list_models.return_value = ["qwen3:8b", "nomic-embed-text"]
    return client


@pytest.fixture
def checker(tmp_path, mock_client):
    """HealthChecker instance with isolated temporary base directory."""
    # Create required directory structure inside tmp_path
    for d in ["logs", "data", "docs", "src", "tests", "scripts"]:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)

    settings = Settings(
        ollama_base_url="http://localhost:11434",
        ollama_llm_model="qwen3:8b",
        ollama_embed_model="nomic-embed-text",
        _env_file=None,
    )
    return HealthChecker(settings=settings, ollama_client=mock_client, base_dir=tmp_path)


class TestHealthChecker:
    """Test suite for individual and aggregated health checks."""

    def test_check_python_version_pass(self, checker, monkeypatch):
        class MockVersion(tuple):
            major = 3
            minor = 11
            micro = 9

        monkeypatch.setattr(sys, "version_info", MockVersion((3, 11, 9)))
        res = checker.check_python_version()
        assert res.status == HealthStatus.PASS

    def test_check_python_version_warning_higher_version(self, checker, monkeypatch):
        class MockVersion(tuple):
            major = 3
            minor = 12
            micro = 1

        monkeypatch.setattr(sys, "version_info", MockVersion((3, 12, 1)))
        res = checker.check_python_version()
        assert res.status == HealthStatus.WARNING

    def test_check_python_version_fail_old_version(self, checker, monkeypatch):
        class MockVersion(tuple):
            major = 3
            minor = 10
            micro = 12

        monkeypatch.setattr(sys, "version_info", MockVersion((3, 10, 12)))
        res = checker.check_python_version()
        assert res.status == HealthStatus.FAIL

    def test_check_virtual_environment(self, checker, monkeypatch):
        monkeypatch.setenv("VIRTUAL_ENV", "/fake/venv")
        res = checker.check_virtual_environment()
        assert res.status == HealthStatus.PASS

        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        monkeypatch.setattr(sys, "prefix", "/system/python")
        monkeypatch.setattr(sys, "base_prefix", "/system/python")
        if hasattr(sys, "real_prefix"):
            monkeypatch.delattr(sys, "real_prefix")
        res2 = checker.check_virtual_environment()
        assert res2.status == HealthStatus.WARNING

    def test_check_dependencies(self, checker):
        res = checker.check_dependencies()
        assert res.status == HealthStatus.PASS
        assert "requests" in res.details["installed"]

    def test_check_configuration(self, checker):
        res = checker.check_configuration()
        assert res.status == HealthStatus.PASS
        assert "ollama_base_url" in res.details

    def test_check_directories_pass(self, checker):
        res = checker.check_directories()
        assert res.status == HealthStatus.PASS

    def test_check_directories_fail(self, tmp_path, mock_client):
        # Empty tmp dir with no subdirectories
        empty_checker = HealthChecker(
            settings=Settings(_env_file=None),
            ollama_client=mock_client,
            base_dir=tmp_path / "empty",
        )
        (tmp_path / "empty").mkdir()
        res = empty_checker.check_directories()
        assert res.status == HealthStatus.FAIL
        assert "logs" in res.details["missing"]

    def test_check_ollama_server_pass(self, checker):
        res = checker.check_ollama_server()
        assert res.status == HealthStatus.PASS

    def test_check_ollama_server_fail(self, checker, mock_client):
        mock_client.is_server_reachable.return_value = False
        res = checker.check_ollama_server()
        assert res.status == HealthStatus.FAIL

    def test_check_llm_model_pass(self, checker, mock_client):
        mock_client.is_model_installed.return_value = True
        res = checker.check_llm_model()
        assert res.status == HealthStatus.PASS

    def test_check_llm_model_missing(self, checker, mock_client):
        mock_client.is_model_installed.return_value = False
        res = checker.check_llm_model()
        assert res.status == HealthStatus.FAIL
        assert "not installed in Ollama" in res.message

    def test_check_embed_model_pass(self, checker, mock_client):
        mock_client.is_model_installed.return_value = True
        res = checker.check_embed_model()
        assert res.status == HealthStatus.PASS

    def test_check_embed_model_missing(self, checker, mock_client):
        mock_client.is_model_installed.return_value = False
        res = checker.check_embed_model()
        assert res.status == HealthStatus.FAIL

    def test_run_all_checks_aggregated(self, checker):
        report = checker.run_all_checks()
        assert report.overall_status in (HealthStatus.PASS, HealthStatus.WARNING)
        assert len(report.checks) == 8
        assert report.is_ready is True
