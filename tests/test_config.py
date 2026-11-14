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

    def test_phase5_graph_retrieval_settings(self):
        """Test Phase 5 graph retrieval configuration defaults and validation."""
        cfg = Settings(_env_file=None)
        assert cfg.graph_max_hops == 2
        assert cfg.graph_max_results == 10
        assert cfg.graph_max_paths == 50
        assert cfg.graph_traversal_direction == "both"
        assert cfg.graph_max_degree == 50

        # Custom valid values
        cfg_custom = Settings(
            graph_max_hops=3,
            graph_traversal_direction="OUT",
            graph_max_results=25,
            _env_file=None,
        )
        assert cfg_custom.graph_max_hops == 3
        assert cfg_custom.graph_traversal_direction == "out"
        assert cfg_custom.graph_max_results == 25

        # Invalid hops (must be 1 to 3)
        with pytest.raises(ValidationError):
            Settings(graph_max_hops=0, _env_file=None)
        with pytest.raises(ValidationError):
            Settings(graph_max_hops=4, _env_file=None)

        # Invalid direction
        with pytest.raises(ValidationError):
            Settings(graph_traversal_direction="sideways", _env_file=None)

        # Non-positive numbers
        with pytest.raises(ValidationError):
            Settings(graph_max_results=0, _env_file=None)
        with pytest.raises(ValidationError):
            Settings(graph_max_paths=-1, _env_file=None)

    def test_phase6_hybrid_retrieval_settings(self):
        """Test Phase 6 hybrid retrieval settings defaults and validation."""
        cfg = Settings(_env_file=None)
        assert cfg.hybrid_top_k == 10
        assert cfg.hybrid_vector_top_k == 10
        assert cfg.hybrid_graph_top_k == 10
        assert cfg.hybrid_rrf_k == 60
        assert cfg.hybrid_vector_weight == 1.0
        assert cfg.hybrid_graph_weight == 1.0
        assert cfg.hybrid_min_rrf_score == 0.0
        assert cfg.hybrid_parallel_retrieval is False

        # Custom valid values
        cfg_custom = Settings(
            hybrid_top_k=5,
            hybrid_rrf_k=30,
            hybrid_vector_weight=0.8,
            hybrid_graph_weight=1.2,
            hybrid_min_rrf_score=0.01,
            hybrid_parallel_retrieval=True,
            _env_file=None,
        )
        assert cfg_custom.hybrid_top_k == 5
        assert cfg_custom.hybrid_rrf_k == 30
        assert cfg_custom.hybrid_vector_weight == 0.8
        assert cfg_custom.hybrid_graph_weight == 1.2
        assert cfg_custom.hybrid_min_rrf_score == 0.01
        assert cfg_custom.hybrid_parallel_retrieval is True

        # Both weights zero must be rejected
        with pytest.raises(ValidationError):
            Settings(hybrid_vector_weight=0.0, hybrid_graph_weight=0.0, _env_file=None)

        # Negative weights rejected
        with pytest.raises(ValidationError):
            Settings(hybrid_vector_weight=-0.5, _env_file=None)
        with pytest.raises(ValidationError):
            Settings(hybrid_graph_weight=-0.1, _env_file=None)

        # Non-positive K values rejected
        with pytest.raises(ValidationError):
            Settings(hybrid_top_k=0, _env_file=None)
        with pytest.raises(ValidationError):
            Settings(hybrid_rrf_k=0, _env_file=None)

    def test_phase7_rag_settings(self):
        """Test Phase 7 grounded RAG settings defaults and validation."""
        cfg = Settings(_env_file=None)
        assert cfg.rag_generation_model == "qwen3:8b"
        assert cfg.rag_temperature == 0.1
        assert cfg.rag_top_k == 5
        assert cfg.rag_max_context_tokens == 3000
        assert cfg.rag_max_generation_tokens == 1024
        assert cfg.rag_generation_timeout == 180
        assert cfg.rag_min_evidence_items == 1
        assert cfg.rag_require_citations is True
        assert cfg.rag_enable_faithfulness_check is True

        # Custom values
        cfg_custom = Settings(
            rag_generation_model="custom-model",
            rag_temperature=0.7,
            rag_top_k=10,
            rag_max_context_tokens=4000,
            rag_max_generation_tokens=2048,
            rag_generation_timeout=240,
            rag_min_evidence_items=2,
            rag_require_citations=False,
            rag_enable_faithfulness_check=False,
            _env_file=None,
        )
        assert cfg_custom.rag_generation_model == "custom-model"
        assert cfg_custom.rag_temperature == 0.7
        assert cfg_custom.rag_top_k == 10
        assert cfg_custom.rag_max_context_tokens == 4000
        assert cfg_custom.rag_max_generation_tokens == 2048
        assert cfg_custom.rag_generation_timeout == 240
        assert cfg_custom.rag_min_evidence_items == 2
        assert cfg_custom.rag_require_citations is False
        assert cfg_custom.rag_enable_faithfulness_check is False

        # Invalid temperature > 2.0 or < 0.0
        with pytest.raises(ValidationError):
            Settings(rag_temperature=-0.1, _env_file=None)
        with pytest.raises(ValidationError):
            Settings(rag_temperature=2.5, _env_file=None)

        # Invalid positive ints
        with pytest.raises(ValidationError):
            Settings(rag_top_k=0, _env_file=None)
        with pytest.raises(ValidationError):
            Settings(rag_max_context_tokens=0, _env_file=None)
        with pytest.raises(ValidationError):
            Settings(rag_max_generation_tokens=0, _env_file=None)
        with pytest.raises(ValidationError):
            Settings(rag_generation_timeout=0, _env_file=None)
        with pytest.raises(ValidationError):
            Settings(rag_min_evidence_items=0, _env_file=None)
