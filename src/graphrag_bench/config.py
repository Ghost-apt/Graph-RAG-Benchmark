"""Configuration management for Graph RAG Benchmark.

Loads and validates environment variables using pydantic-settings and python-dotenv.
Ensures sensitive fields are masked when serialized or represented as strings.
"""

from functools import lru_cache
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Application and infrastructure configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Ollama settings
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for the local Ollama HTTP REST server",
    )
    ollama_llm_model: str = Field(
        default="qwen3:8b",
        description="Ollama model identifier for text generation and QA reasoning",
    )
    ollama_embed_model: str = Field(
        default="nomic-embed-text",
        description="Ollama model identifier for vector embedding generation",
    )
    ollama_timeout: int = Field(
        default=120,
        description="Request timeout in seconds for Ollama API calls",
    )

    # Logging settings
    log_level: LogLevel = Field(
        default="INFO",
        description="Standard logging verbosity level",
    )
    log_file: str = Field(
        default="logs/graphrag.log",
        description="Path to persistent log file on disk",
    )

    # Application settings
    app_env: str = Field(
        default="development",
        description="Runtime environment identifier (e.g., development, testing, production)",
    )

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_base_url(cls, v: str) -> str:
        """Validate and normalize Ollama base URL."""
        v = v.strip()
        if not v:
            raise ValueError("OLLAMA_BASE_URL cannot be empty.")
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"OLLAMA_BASE_URL must start with http:// or https://, got: {v}")
        return v.rstrip("/")

    @field_validator("ollama_timeout")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """Validate timeout is a positive integer."""
        if v <= 0:
            raise ValueError(f"OLLAMA_TIMEOUT must be greater than 0, got: {v}")
        return v

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, v: Any) -> str:
        """Convert log level string to uppercase."""
        if isinstance(v, str):
            v_upper = v.strip().upper()
            valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
            if v_upper not in valid_levels:
                raise ValueError(
                    f"Invalid LOG_LEVEL '{v}'. Must be one of: {', '.join(sorted(valid_levels))}"
                )
            return v_upper
        return v

    def to_safe_dict(self) -> dict[str, Any]:
        """Return configuration dictionary with sensitive fields safely sanitized.

        Guarantees that credentials, tokens, or keys are never exposed in logs or dumps.
        """
        data = self.model_dump()
        sensitive_keywords = {"key", "secret", "token", "password", "auth", "credential"}
        for k in data:
            if any(kw in k.lower() for kw in sensitive_keywords):
                data[k] = "******"
        return data

    def __repr__(self) -> str:
        """Return safe string representation concealing secrets."""
        safe_items = [f"{k}={v!r}" for k, v in self.to_safe_dict().items()]
        return f"Settings({', '.join(safe_items)})"

    def __str__(self) -> str:
        return repr(self)


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton instance of validated application settings."""
    return Settings()


def reload_settings() -> Settings:
    """Clear cache and reload settings from environment and .env."""
    get_settings.cache_clear()
    return get_settings()
