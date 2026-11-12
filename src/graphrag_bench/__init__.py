"""Graph RAG Benchmark - Core Package.

Phase 1: Environment Setup & Project Foundation.
"""

from graphrag_bench.config import Settings, get_settings
from graphrag_bench.health import HealthChecker, HealthReport, HealthStatus
from graphrag_bench.logger import get_logger, setup_logger
from graphrag_bench.ollama_client import (
    GenerateResponse,
    OllamaAPIError,
    OllamaClient,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
)

__version__ = "0.1.0"
__all__ = [
    "GenerateResponse",
    "HealthChecker",
    "HealthReport",
    "HealthStatus",
    "OllamaAPIError",
    "OllamaClient",
    "OllamaConnectionError",
    "OllamaModelNotFoundError",
    "OllamaTimeoutError",
    "Settings",
    "__version__",
    "get_logger",
    "get_settings",
    "setup_logger",
]
