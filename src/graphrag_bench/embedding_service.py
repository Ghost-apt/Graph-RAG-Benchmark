"""Vector embedding service for Graph RAG Benchmark.

Reuses the core OllamaClient to compute high-dimensional semantic vector embeddings
using local models (default: nomic-embed-text) with:
- Latency profiling and metrics
- Numerical validity checks (finite floats, non-empty)
- Dynamic dimension discovery and consistency enforcement
- Controlled batching for large document corpora
- Robust exception handling wrapping Ollama error taxonomy
"""

import math
import time
from dataclasses import dataclass

from graphrag_bench.config import Settings, get_settings
from graphrag_bench.logger import get_logger
from graphrag_bench.ollama_client import (
    OllamaAPIError,
    OllamaClient,
    OllamaConnectionError,
    OllamaError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
)

logger = get_logger("embedding_service")


class EmbeddingError(Exception):
    """Base exception for embedding generation failures."""


class EmptyTextError(ValueError, EmbeddingError):
    """Raised when text provided for embedding is empty or whitespace-only."""


class DimensionMismatchError(EmbeddingError):
    """Raised when vectors within a batch have inconsistent dimensions."""


class InvalidVectorError(EmbeddingError):
    """Raised when vectors contain non-finite numbers, NaNs, or incorrect types."""


@dataclass
class EmbeddingResult:
    """Container for batch or single embedding generation outcomes."""

    vectors: list[list[float]]
    model: str
    dimensions: int
    duration_ms: float
    input_count: int


class EmbeddingService:
    """High-level semantic embedding service delegating to OllamaClient."""

    def __init__(
        self,
        client: OllamaClient | None = None,
        settings: Settings | None = None,
        ollama_client: OllamaClient | None = None,
    ) -> None:
        """Initialize embedding service.

        Args:
            client: Optional OllamaClient instance. Reuses existing client if supplied.
            settings: Optional application settings.
            ollama_client: Alias for client parameter.
        """
        self.settings = settings or get_settings()
        effective_client = client or ollama_client or OllamaClient(settings=self.settings)
        self.client = effective_client
        self.default_model = self.settings.ollama_embed_model
        self.default_batch_size = self.settings.embedding_batch_size

    def embed_text(
        self,
        text: str,
        model: str | None = None,
    ) -> list[float]:
        """Generate a single normalized dense vector embedding.

        Args:
            text: Input string to embed.
            model: Optional Ollama embedding model override.

        Returns:
            List of floats representing the dense semantic vector.

        Raises:
            EmptyTextError: If text is empty or blank.
            EmbeddingError: If Ollama API fails or vector is malformed.
        """
        if not text or not text.strip():
            raise EmptyTextError("Cannot generate embedding for empty or whitespace-only text.")

        result = self.embed_batch([text], model=model, batch_size=1)
        return result.vectors[0]

    def embed_batch(
        self,
        texts: list[str],
        model: str | None = None,
        batch_size: int | None = None,
    ) -> EmbeddingResult:
        """Generate dense vector embeddings for a collection of texts in batches.

        Args:
            texts: List of input text strings.
            model: Optional embedding model override.
            batch_size: Optional batch size override (defaults to settings.embedding_batch_size).

        Returns:
            EmbeddingResult with all vectors, discovered dimensions, and latency in ms.

        Raises:
            EmptyTextError: If texts list is empty or any individual item is empty.
            EmbeddingError: If Ollama fails, returns malformed vectors, or dimensions mismatch.
        """
        if not texts:
            raise EmptyTextError("Input texts list for embedding cannot be empty.")

        for idx, t in enumerate(texts):
            if not t or not t.strip():
                raise EmptyTextError(
                    f"Item at index {idx} in texts batch is empty or whitespace-only."
                )

        target_model = model or self.default_model
        bsize = max(1, batch_size or self.default_batch_size)
        all_vectors: list[list[float]] = []
        discovered_dims: int | None = None

        start_time = time.perf_counter()

        logger.debug(
            "Generating embeddings for %d texts with model '%s' (batch_size=%d)",
            len(texts),
            target_model,
            bsize,
        )

        try:
            for offset in range(0, len(texts), bsize):
                chunk = texts[offset : offset + bsize]
                embed_resp = self.client.embed(chunk, model=target_model)
                batch_vectors = embed_resp.embeddings

                for v in batch_vectors:
                    dims = self._validate_numerical_vector(v)
                    if discovered_dims is None:
                        discovered_dims = dims
                    elif dims != discovered_dims:
                        raise DimensionMismatchError(
                            f"Inconsistent vector dimensions: expected {discovered_dims}, got {dims}"
                        )
                    all_vectors.append(v)

        except (
            OllamaModelNotFoundError,
            OllamaConnectionError,
            OllamaTimeoutError,
            OllamaAPIError,
        ) as exc:
            logger.error("Ollama error during embedding generation: %s", exc)
            raise EmbeddingError(f"Embedding generation failed: {exc}") from exc
        except OllamaError as exc:
            logger.error("General Ollama error: %s", exc)
            raise EmbeddingError(f"Ollama failure: {exc}") from exc

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        if discovered_dims is None:
            raise EmbeddingError("Failed to discover vector dimensions: no embeddings generated.")

        logger.info(
            "Generated %d embeddings using '%s' (%d dims) in %.1fms",
            len(all_vectors),
            target_model,
            discovered_dims,
            duration_ms,
        )

        return EmbeddingResult(
            vectors=all_vectors,
            model=target_model,
            dimensions=discovered_dims,
            duration_ms=duration_ms,
            input_count=len(texts),
        )

    def _validate_numerical_vector(self, vector: list[float]) -> int:
        """Validate that a vector contains finite float numbers and return its dimension."""
        if not vector:
            raise InvalidVectorError("Vector is empty.")

        for val in vector:
            if not isinstance(val, (int, float)):
                raise InvalidVectorError(f"Vector component is not a number: {val!r}")
            if math.isnan(val) or math.isinf(val):
                raise InvalidVectorError(f"Vector component is non-finite (NaN or Inf): {val}")

        return len(vector)
