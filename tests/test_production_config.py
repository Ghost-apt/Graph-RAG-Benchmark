"""Unit tests for Phase 9 production configuration and security validation."""

import pytest
from pydantic import ValidationError

from graphrag_bench.config import Settings


class TestProductionConfig:
    """Test suite for Phase 9 production settings and path traversal protection."""

    def test_production_default_settings(self):
        """Verify Phase 9 configuration parameters defaults."""
        cfg = Settings(_env_file=None)
        assert cfg.app_host == "127.0.0.1"
        assert cfg.app_port == 8501
        assert cfg.benchmark_dir == "data/phase8_benchmark"
        assert cfg.backup_dir == "data/backups"
        assert cfg.log_format == "text"
        assert cfg.log_max_bytes == 10485760
        assert cfg.log_backup_count == 5
        assert cfg.max_concurrent_generations == 2
        assert cfg.max_concurrent_benchmarks == 1
        assert cfg.auth_mode == "none"
        assert cfg.enable_telemetry is True

    def test_custom_production_settings(self, monkeypatch):
        """Verify overriding Phase 9 settings via environment variables."""
        monkeypatch.setenv("APP_HOST", "0.0.0.0")
        monkeypatch.setenv("APP_PORT", "9000")
        monkeypatch.setenv("LOG_FORMAT", "json")
        monkeypatch.setenv("MAX_CONCURRENT_GENERATIONS", "4")
        monkeypatch.setenv("AUTH_MODE", "basic")

        cfg = Settings(_env_file=None)
        assert cfg.app_host == "0.0.0.0"
        assert cfg.app_port == 9000
        assert cfg.log_format == "json"
        assert cfg.max_concurrent_generations == 4
        assert cfg.auth_mode == "basic"

    def test_path_traversal_rejection(self):
        """Verify that path traversal sequences ('..') in directory settings raise ValidationError."""
        with pytest.raises(ValidationError):
            Settings(data_dir="../data", _env_file=None)

        with pytest.raises(ValidationError):
            Settings(backup_dir="data/../../etc/passwd", _env_file=None)

        with pytest.raises(ValidationError):
            Settings(processed_dir="data/../processed", _env_file=None)

    def test_port_and_concurrency_bounds(self):
        """Verify invalid port and concurrency bounds raise ValidationError."""
        with pytest.raises(ValidationError):
            Settings(app_port=0, _env_file=None)

        with pytest.raises(ValidationError):
            Settings(app_port=70000, _env_file=None)

        with pytest.raises(ValidationError):
            Settings(max_concurrent_generations=0, _env_file=None)

        with pytest.raises(ValidationError):
            Settings(max_concurrent_generations=20, _env_file=None)
