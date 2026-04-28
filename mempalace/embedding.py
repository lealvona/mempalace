"""Embedding function factory with hardware acceleration and remote endpoint support.

Returns a ChromaDB-compatible embedding function. Two backends are supported:

**Remote endpoint** (preferred when configured):
    When ``EMBED_REMOTE_URL`` env var or ``embedding_remote_url`` in
    ``~/.mempalace/config.json`` is set, embeddings are computed by a
    centralized OpenAI-compatible ``/v1/embeddings`` server (e.g. an
    Ollama instance running ``nomic-embed-text-v1.5``). This avoids
    loading ONNX locally and lets a fleet share one GPU inference host.

**Local ONNX** (default fallback):
    Uses ``all-MiniLM-L6-v2`` (384 dims) via
    ``chromadb.utils.embedding_functions.ONNXMiniLM_L6_V2`` with a
    user-selected ONNX Runtime execution provider. Supported devices
    (env ``MEMPALACE_EMBEDDING_DEVICE`` or ``embedding_device`` in config):

    * ``auto`` — prefer CUDA ▸ CoreML ▸ DirectML, fall back to CPU
    * ``cpu`` — force CPU (the historical default)
    * ``cuda`` — NVIDIA GPU via ``onnxruntime-gpu``
    * ``coreml`` — Apple Neural Engine (macOS)
    * ``dml`` — DirectML (Windows / AMD / Intel GPUs)

Requesting an unavailable accelerator emits a warning and falls back to CPU
rather than hard-failing — mining must still work on a laptop without CUDA.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_PROVIDER_MAP = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
    "dml": ["DmlExecutionProvider", "CPUExecutionProvider"],
}

_DEVICE_EXTRA = {
    "cuda": "mempalace[gpu]",
    "coreml": "mempalace[coreml]",
    "dml": "mempalace[dml]",
}

_AUTO_ORDER = [
    ("CUDAExecutionProvider", "cuda"),
    ("CoreMLExecutionProvider", "coreml"),
    ("DmlExecutionProvider", "dml"),
]

_EF_CACHE: dict = {}
_WARNED: set = set()

# ── Remote embedding support ─────────────────────────────────────────────

_DEFAULT_REMOTE_MODEL = "nomic-embed-text-v1.5"
_REMOTE_EF_CACHE_KEY = "__remote__"


class RemoteEmbeddingFunction:
    """ChromaDB-compatible embedding function backed by a remote
    OpenAI-compatible ``/v1/embeddings`` endpoint.

    Designed for fleet deployments where a single GPU host (e.g. Ollama)
    serves ``nomic-embed-text-v1.5`` embeddings to many MemPalace workers.

    Parameters
    ----------
    base_url:
        Root URL of the embedding server, e.g. ``"http://192.168.1.125:11435"``.
    model:
        Model name the server should use (default ``"nomic-embed-text-v1.5"``).
    timeout:
        Per-request HTTP timeout in seconds (default 60).
    """

    def __init__(
        self,
        base_url: str,
        model: str = _DEFAULT_REMOTE_MODEL,
        timeout: int = 60,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        # Lazy-import so the module loads fine without requests present
        # (will error only when actually called).
        try:
            import requests  # noqa: F401

            self._requests = requests
        except ImportError as exc:
            raise ImportError(
                "The 'requests' library is required for remote embeddings. "
                "Install it with: pip install requests"
            ) from exc
        logger.info(
            "RemoteEmbeddingFunction configured: url=%s model=%s",
            self._base_url,
            self._model,
        )

    # -- ChromaDB contract --------------------------------------------------

    @staticmethod
    def name() -> str:
        """Return ``'default'`` for ChromaDB collection compatibility."""
        return "default"

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Embed a batch of strings and return float vectors.

        Uses a single HTTP request for the whole batch (the OpenAI API
        accepts a list of strings). Falls back to per-item requests only
        if the batch endpoint returns an unexpected shape.
        """
        if not input:
            return []

        url = f"{self._base_url}/v1/embeddings"
        payload = {"model": self._model, "input": input}

        try:
            resp = self._requests.post(
                url,
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
        except self._requests.exceptions.ConnectionError as exc:
            logger.error(
                "Remote embedding server unreachable at %s — %s",
                url,
                exc,
            )
            raise
        except self._requests.exceptions.Timeout:
            logger.error(
                "Remote embedding server timed out after %ds (%s)",
                self._timeout,
                url,
            )
            raise
        except self._requests.exceptions.HTTPError as exc:
            logger.error(
                "Remote embedding server returned HTTP %s: %s",
                resp.status_code,
                exc,
            )
            raise

        body = resp.json()
        data = body.get("data")

        # OpenAI /v1/embeddings shape: {"data": [{"embedding": [...]}], ...}
        if isinstance(data, list) and len(data) == len(input):
            embeddings: list[list[float]] = []
            for item in data:
                emb = item.get("embedding") if isinstance(item, dict) else None
                if not isinstance(emb, list):
                    logger.error(
                        "Unexpected embedding response shape: %s",
                        type(item),
                    )
                    raise ValueError(
                        f"Remote server returned unexpected embedding data: {item!r}"
                    )
                embeddings.append(emb)
            return embeddings

        # If the batch shape is unexpected, try per-item as a fallback
        logger.warning(
            "Batch embedding response shape mismatch (got %d items for %d inputs), "
            "falling back to per-item requests",
            len(data) if isinstance(data, list) else -1,
            len(input),
        )
        results: list[list[float]] = []
        for text in input:
            single = self._embed_single(text)
            results.append(single)
        return results

    # -- internals ----------------------------------------------------------

    def _embed_single(self, text: str) -> list[float]:
        """Embed a single string, raising on failure."""
        url = f"{self._base_url}/v1/embeddings"
        payload = {"model": self._model, "input": [text]}
        resp = self._requests.post(url, json=payload, timeout=self._timeout)
        resp.raise_for_status()
        body = resp.json()
        return body["data"][0]["embedding"]


def _resolve_remote_url() -> Optional[str]:
    """Return the configured remote embedding URL, or ``None``.

    Resolution order:
    1. ``EMBED_REMOTE_URL`` environment variable
    2. ``embedding_remote_url`` in ``~/.mempalace/config.json``
    """
    env_val = os.environ.get("EMBED_REMOTE_URL")
    if env_val and env_val.strip():
        return env_val.strip()

    # Try config file directly (avoids circular import with .config)
    config_path = os.path.expanduser("~/.mempalace/config.json")
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
        url = cfg.get("embedding_remote_url")
        if isinstance(url, str) and url.strip():
            return url.strip()
    except (OSError, json.JSONDecodeError):
        pass

    return None


def _resolve_providers(device: str) -> tuple[list, str]:
    """Return ``(provider_list, effective_device)`` for ``device``.

    Falls back to CPU (with a one-shot warning) when the requested
    accelerator is not compiled into the installed ``onnxruntime``.
    """
    device = (device or "auto").strip().lower()

    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
    except ImportError:
        return (["CPUExecutionProvider"], "cpu")

    if device == "auto":
        for provider, name in _AUTO_ORDER:
            if provider in available:
                return ([provider, "CPUExecutionProvider"], name)
        return (["CPUExecutionProvider"], "cpu")

    requested = _PROVIDER_MAP.get(device)
    if requested is None:
        if device not in _WARNED:
            logger.warning("Unknown embedding_device %r — falling back to cpu", device)
            _WARNED.add(device)
        return (["CPUExecutionProvider"], "cpu")

    preferred = requested[0]
    if preferred == "CPUExecutionProvider":
        return (requested, "cpu")

    if preferred not in available:
        if device not in _WARNED:
            extra = _DEVICE_EXTRA.get(device, "the matching mempalace extra for your device")
            logger.warning(
                "embedding_device=%r requested but %s is not installed — "
                "falling back to CPU. Install %s.",
                device,
                preferred,
                extra,
            )
            _WARNED.add(device)
        return (["CPUExecutionProvider"], "cpu")

    return (requested, device)


def _build_ef_class():
    """Subclass ``ONNXMiniLM_L6_V2`` with name ``"default"``.

    Why the rename: ChromaDB 1.5 persists the EF identity on the collection
    and rejects reads that pass a differently-named EF (``onnx_mini_lm_l6_v2``
    vs ``default``). The vectors and model are identical — only the
    ``name()`` tag differs — so spoofing the name lets one EF class serve
    palaces created with ``DefaultEmbeddingFunction`` *and* palaces we
    create ourselves, with the same GPU-capable ``preferred_providers``.
    """
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    class _MempalaceONNX(ONNXMiniLM_L6_V2):
        @staticmethod
        def name() -> str:
            return "default"

    return _MempalaceONNX


def get_embedding_function(device: Optional[str] = None):
    """Return a cached embedding function.

    If a remote embedding URL is configured (``EMBED_REMOTE_URL`` env var
    or ``embedding_remote_url`` in config.json), returns a
    :class:`RemoteEmbeddingFunction` that talks to the centralized server.
    Otherwise falls back to local ONNX with the requested device.

    ``device=None`` reads from :class:`MempalaceConfig.embedding_device`.
    The returned function is shared across calls with the same resolved
    provider list (or remote URL) so we only pay model-load cost once
    per process.
    """
    # ── Remote endpoint takes priority when configured ──
    remote_url = _resolve_remote_url()
    if remote_url:
        cached = _EF_CACHE.get(_REMOTE_EF_CACHE_KEY)
        if cached is not None:
            return cached
        try:
            ef = RemoteEmbeddingFunction(base_url=remote_url)
            _EF_CACHE[_REMOTE_EF_CACHE_KEY] = ef
            logger.info("Using remote embedding endpoint: %s", remote_url)
            return ef
        except Exception:
            logger.warning(
                "Failed to initialise remote embedding function at %s — "
                "falling back to local ONNX",
                remote_url,
                exc_info=True,
            )
            # Deliberate fall-through to local ONNX below

    # ── Local ONNX fallback ──
    if device is None:
        from .config import MempalaceConfig

        device = MempalaceConfig().embedding_device

    providers, effective = _resolve_providers(device)
    cache_key = ("local", tuple(providers))
    cached = _EF_CACHE.get(cache_key)
    if cached is not None:
        return cached

    ef_cls = _build_ef_class()
    ef = ef_cls(preferred_providers=providers)
    _EF_CACHE[cache_key] = ef
    logger.info("Embedding function initialized (device=%s providers=%s)", effective, providers)
    return ef


def describe_device(device: Optional[str] = None) -> str:
    """Return a short human-readable label for the resolved device.

    Used by the miner CLI header so users can see at a glance whether GPU
    acceleration actually engaged. Returns ``'remote'`` when the remote
    embedding endpoint is configured.
    """
    remote_url = _resolve_remote_url()
    if remote_url:
        return "remote"
    if device is None:
        from .config import MempalaceConfig

        device = MempalaceConfig().embedding_device
    _, effective = _resolve_providers(device)
    return effective
