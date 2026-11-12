"""Unit tests for configuration management (graphrag_bench.config)."""

import pytest
from pydantic import ValidationError

from graphrag_bench.config import Settings, get_settings, reload_settings


class TestConfig:
    """Test suite for configuration validation and loading."""

    def test_default_settings(self, monkeypatch):
        """Test default values when no environment variables are set."""
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_LLM_MODEL", raising=False)
        monkeypatch.delenv("OLLAMA_EMBED_MODEL", raising=False)
        monkeypatch.delenv("OLLAMA_TIMEOUT", raising=False)
        monkeypatch.delenv("LOG_LEVEL", raising=False)

        cfg = Settings(_env_file=None)
        assert cfg.ollama_base_url == "http://localhost:11434"
        assert cfg.ollama_llm_model == "qwen3:8b"
        assert cfg.ollama_embed_model == "nomic-embed-text"
        assert cfg.ollama_timeout == 120
        assert cfg.log_level == "INFO"
        assert cfg.app_env == "development"

    def test_custom_environment_override(self, monkeypatch):
        """Test overriding settings via environment variables."""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://192.168.1.50:11434/")
        monkeypatch.setenv("OLLAMA_LLM_MODEL", "custom-llm:14b")
        monkeypatch.setenv("OLLAMA_EMBED_MODEL", "custom-embed")
        monkeypatch.setenv("OLLAMA_TIMEOUT", "60")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")

        cfg = Settings(_env_file=None)
        # Trailing slash should be automatically stripped
        assert cfg.ollama_base_url == "http://192.168.1.50:11434"
        assert cfg.ollama_llm_model == "custom-llm:14b"
        assert cfg.ollama_embed_model == "custom-embed"
        assert cfg.ollama_timeout == 60
        assert cfg.log_level == "DEBUG"

    def test_invalid_base_url(self):
        """Test that invalid URLs without http:// or https:// raise ValidationError."""
        with pytest.raises(ValidationError):
            Settings(ollama_base_url="localhost:11434", _env_file=None)

        with pytest.raises(ValidationError):
            Settings(ollama_base_url="", _env_file=None)

    def test_invalid_timeout(self):
        """Test that non-positive timeouts raise ValidationError."""
        with pytest.raises(ValidationError):
            Settings(ollama_timeout=0, _env_file=None)

        with pytest.raises(ValidationError):
            Settings(ollama_timeout=-10, _env_file=None)

    def test_log_level_normalization(self):
        """Test case-insensitive log level normalization."""
        cfg = Settings(log_level="warning", _env_file=None)
        assert cfg.log_level == "WARNING"

        cfg2 = Settings(log_level="debug", _env_file=None)
        assert cfg2.log_level == "DEBUG"

    def test_invalid_log_level(self):
        """Test that unknown log levels raise ValidationError."""
        with pytest.raises(ValidationError):
            Settings(log_level="SUPER_VERBOSE", _env_file=None)

    def test_safe_dict_masking(self):
        """Test that sensitive keywords are masked in to_safe_dict and repr."""
        cfg = Settings(
            ollama_base_url="http://localhost:11434",
            _env_file=None,
        )
        safe = cfg.to_safe_dict()
        assert "ollama_base_url" in safe
        assert "ollama_llm_model" in safe

        # Representation must not leak raw secrets
        rep = repr(cfg)
        assert "Settings(" in rep
        assert "ollama_base_url='http://localhost:11434'" in rep

    def test_singleton_and_reload(self, monkeypatch):
        """Test get_settings caching and reload_settings."""
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

        monkeypatch.setenv("APP_ENV", "testing_reload")
        s3 = reload_settings()
        assert s3.app_env == "testing_reload"
