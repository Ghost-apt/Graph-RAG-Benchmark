"""Ollama HTTP REST Client for Graph RAG Benchmark.

Provides resilient integration with local Ollama instances, handling:
- Server reachability and health checks
- Model catalog inspection and presence verification
- Synchronous text generation (/api/generate)
- High-performance vector embeddings (/api/embed)
- Robust error classification (connection, timeout, model missing, malformed data)
"""

from dataclasses import dataclass, field
from typing import Any

import requests
from requests.exceptions import ConnectionError as ReqConnectionError
from requests.exceptions import HTTPError, RequestException
from requests.exceptions import Timeout as ReqTimeout

from graphrag_bench.config import Settings, get_settings
from graphrag_bench.logger import get_logger

logger = get_logger("ollama_client")


class OllamaError(Exception):
    """Base exception for all Ollama client errors."""


class OllamaConnectionError(OllamaError):
    """Raised when the Ollama server cannot be reached."""


class OllamaTimeoutError(OllamaError):
    """Raised when an Ollama API operation times out."""


class OllamaModelNotFoundError(OllamaError):
    """Raised when a requested model is not available on the Ollama host."""


class OllamaAPIError(OllamaError):
    """Raised when the Ollama server returns an unexpected error or malformed payload."""


@dataclass
class GenerateResponse:
    """Structured response container for text generation."""

    model: str
    response: str
    done: bool = True
    total_duration_ns: int | None = None
    eval_count: int | None = None
    prompt_eval_count: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingResponse:
    """Structured response container for vector embeddings."""

    model: str
    embeddings: list[list[float]]
    dimensions: int
    raw: dict[str, Any] = field(default_factory=dict)


class OllamaClient:
    """Synchronous HTTP client for interacting with the Ollama REST API."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int | None = None,
        default_llm: str | None = None,
        default_embed: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize OllamaClient with explicit parameters or environment settings.

        Args:
            base_url: Base URL of the Ollama server (e.g., 'http://localhost:11434').
            timeout: Request timeout in seconds.
            default_llm: Default model identifier for text generation.
            default_embed: Default model identifier for embeddings.
            settings: Optional Settings instance.
        """
        cfg = settings or get_settings()
        self.base_url = (base_url or cfg.ollama_base_url).rstrip("/")
        self.timeout = timeout if timeout is not None else cfg.ollama_timeout
        self.default_llm = default_llm or cfg.ollama_llm_model
        self.default_embed = default_embed or cfg.ollama_embed_model
        self.session = requests.Session()

    def is_server_reachable(self) -> bool:
        """Check if the Ollama server is responsive.

        Returns:
            True if server responds with HTTP 200 to /api/tags, False otherwise.
        """
        try:
            url = f"{self.base_url}/api/tags"
            resp = self.session.get(url, timeout=min(self.timeout, 5))
            return resp.status_code == 200
        except RequestException as exc:
            logger.debug("Ollama server check failed: %s", exc)
            return False

    def list_models(self) -> list[str]:
        """Retrieve all installed model tags from the Ollama host.

        Returns:
            List of model names (e.g. ['qwen3:8b', 'nomic-embed-text:latest']).

        Raises:
            OllamaConnectionError: If server is unreachable.
            OllamaTimeoutError: If request times out.
            OllamaAPIError: If response is malformed or invalid HTTP status.
        """
        url = f"{self.base_url}/api/tags"
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict) or "models" not in data:
                raise OllamaAPIError(
                    f"Malformed /api/tags response: missing 'models' key in {data}"
                )
            models = [m.get("name", "") for m in data["models"] if isinstance(m, dict)]
            return [m for m in models if m]
        except ReqConnectionError as exc:
            msg = f"Failed to connect to Ollama server at {self.base_url}: {exc}"
            logger.error(msg)
            raise OllamaConnectionError(msg) from exc
        except ReqTimeout as exc:
            msg = f"Timed out connecting to Ollama server at {self.base_url} after {self.timeout}s"
            logger.error(msg)
            raise OllamaTimeoutError(msg) from exc
        except HTTPError as exc:
            msg = f"Ollama /api/tags returned HTTP {resp.status_code}: {resp.text}"
            logger.error(msg)
            raise OllamaAPIError(msg) from exc
        except ValueError as exc:
            msg = f"Failed to parse JSON response from Ollama /api/tags: {exc}"
            logger.error(msg)
            raise OllamaAPIError(msg) from exc

    def is_model_installed(self, model_name: str) -> bool:
        """Check if a specific model is installed on the Ollama host.

        Supports standard tag normalization (e.g. matching 'qwen3:8b' with 'qwen3:8b:latest').

        Args:
            model_name: Target model name or tag.

        Returns:
            True if present, False otherwise.
        """
        try:
            installed = self.list_models()
        except OllamaError as exc:
            logger.warning("Could not list models to check '%s': %s", model_name, exc)
            return False

        target = model_name.strip().lower()
        installed_lower = [m.lower() for m in installed]

        # Exact match
        if target in installed_lower:
            return True

        # Normalized match: target without tag matches installed:latest
        if ":" not in target and f"{target}:latest" in installed_lower:
            return True

        # Target with :latest matches installed without :latest
        if target.endswith(":latest") and target[:-7] in installed_lower:
            return True

        # Prefix match for repository names
        for m in installed_lower:
            if m.startswith(f"{target}:") or (":" not in target and m.split(":")[0] == target):
                return True

        return False

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> GenerateResponse:
        """Generate text from a model using the official /api/generate endpoint.

        Args:
            prompt: Text prompt input for the model.
            model: Target model name (defaults to configured LLM).
            system: Optional system instruction prompt.
            options: Optional runtime inference hyperparameters (e.g. temperature).

        Returns:
            GenerateResponse instance with generated text and metadata.

        Raises:
            OllamaModelNotFoundError: If requested model is not installed.
            OllamaConnectionError: If server connection fails.
            OllamaTimeoutError: If generation exceeds timeout.
            OllamaAPIError: On unexpected HTTP or JSON parse errors.
        """
        target_model = model or self.default_llm
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        url = f"{self.base_url}/api/generate"
        payload: dict[str, Any] = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options

        logger.debug("Submitting generation request to %s (model: %s)", url, target_model)
        try:
            resp = self.session.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 404:
                msg = f"Model '{target_model}' not found on Ollama server at {self.base_url}."
                logger.error(msg)
                raise OllamaModelNotFoundError(msg)

            resp.raise_for_status()
            data = resp.json()

            if not isinstance(data, dict) or "response" not in data:
                raise OllamaAPIError(
                    f"Malformed /api/generate response: missing 'response' field in {data}"
                )

            return GenerateResponse(
                model=data.get("model", target_model),
                response=data["response"],
                done=data.get("done", True),
                total_duration_ns=data.get("total_duration"),
                eval_count=data.get("eval_count"),
                prompt_eval_count=data.get("prompt_eval_count"),
                raw=data,
            )
        except ReqConnectionError as exc:
            msg = f"Failed to connect to Ollama server at {self.base_url}: {exc}"
            logger.error(msg)
            raise OllamaConnectionError(msg) from exc
        except ReqTimeout as exc:
            msg = f"Generation request timed out after {self.timeout}s (model: {target_model})"
            logger.error(msg)
            raise OllamaTimeoutError(msg) from exc
        except HTTPError as exc:
            msg = f"Ollama generation failed with HTTP {resp.status_code}: {resp.text}"
            logger.error(msg)
            raise OllamaAPIError(msg) from exc
        except ValueError as exc:
            msg = f"Failed to parse JSON response from /api/generate: {exc}"
            logger.error(msg)
            raise OllamaAPIError(msg) from exc

    def embed(
        self,
        texts: str | list[str],
        model: str | None = None,
    ) -> EmbeddingResponse:
        """Generate dense vector embeddings using the official /api/embed endpoint.

        Args:
            texts: Single text string or list of text strings to embed.
            model: Target embedding model (defaults to configured embed model).

        Returns:
            EmbeddingResponse containing 2D list of float vectors and dimension count.

        Raises:
            OllamaModelNotFoundError: If requested model is not installed.
            OllamaConnectionError: If server connection fails.
            OllamaTimeoutError: If embedding generation exceeds timeout.
            OllamaAPIError: On malformed response, wrong dimensions, or HTTP error.
        """
        target_model = model or self.default_embed
        input_data: str | list[str]
        if isinstance(texts, str):
            if not texts.strip():
                raise ValueError("Embedding input text cannot be empty.")
            input_data = texts
        elif isinstance(texts, list):
            if not texts:
                raise ValueError("Embedding input list cannot be empty.")
            input_data = texts
        else:
            raise TypeError(f"texts must be str or list of str, got {type(texts)}")

        url = f"{self.base_url}/api/embed"
        payload = {
            "model": target_model,
            "input": input_data,
        }

        logger.debug("Submitting embedding request to %s (model: %s)", url, target_model)
        try:
            resp = self.session.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 404:
                msg = f"Embedding model '{target_model}' not found on Ollama server at {self.base_url}."
                logger.error(msg)
                raise OllamaModelNotFoundError(msg)

            resp.raise_for_status()
            data = resp.json()

            if not isinstance(data, dict) or "embeddings" not in data:
                raise OllamaAPIError(
                    f"Malformed /api/embed response: missing 'embeddings' key in {data}"
                )

            embeddings = data["embeddings"]
            if not isinstance(embeddings, list) or len(embeddings) == 0:
                raise OllamaAPIError(
                    "Embedding response contained empty or invalid embeddings list."
                )

            # Validate dimensions and float types
            expected_dim = len(embeddings[0])
            if expected_dim == 0:
                raise OllamaAPIError("Returned embedding vector has zero dimensions.")

            for idx, vec in enumerate(embeddings):
                if not isinstance(vec, list):
                    raise OllamaAPIError(f"Embedding vector at index {idx} is not a list.")
                if len(vec) != expected_dim:
                    raise OllamaAPIError(
                        f"Inconsistent embedding dimensions: index 0 has {expected_dim}, "
                        f"index {idx} has {len(vec)}"
                    )
                if not all(isinstance(v, (int, float)) for v in vec):
                    raise OllamaAPIError(
                        f"Non-numeric values found in embedding vector at index {idx}."
                    )

            return EmbeddingResponse(
                model=data.get("model", target_model),
                embeddings=embeddings,
                dimensions=expected_dim,
                raw=data,
            )
        except ReqConnectionError as exc:
            msg = f"Failed to connect to Ollama server at {self.base_url}: {exc}"
            logger.error(msg)
            raise OllamaConnectionError(msg) from exc
        except ReqTimeout as exc:
            msg = f"Embedding request timed out after {self.timeout}s (model: {target_model})"
            logger.error(msg)
            raise OllamaTimeoutError(msg) from exc
        except HTTPError as exc:
            msg = f"Ollama embedding failed with HTTP {resp.status_code}: {resp.text}"
            logger.error(msg)
            raise OllamaAPIError(msg) from exc
        except ValueError as exc:
            msg = f"Failed to parse JSON response from /api/embed: {exc}"
            logger.error(msg)
            raise OllamaAPIError(msg) from exc

    def embed_single(
        self,
        text: str,
        model: str | None = None,
    ) -> list[float]:
        """Convenience method returning a single 1D vector of floats for a string input.

        Args:
            text: Text to embed.
            model: Optional model identifier.

        Returns:
            List of floats representing the embedding vector.
        """
        res = self.embed(texts=text, model=model)
        return res.embeddings[0]
