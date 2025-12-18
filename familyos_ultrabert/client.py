"""
FamilyOS UltraBERT Client v2.0.1
================================

Production-ready client with automatic warmup for consistent low-latency inference.
Users never see the 285ms cold-start spike - first real call is fast.

Features:
- Automatic model warmup on initialization
- Connection pooling ready
- Batch inference support
- Async-ready design
- Built-in latency tracking
- Graceful error handling

Usage:
    from familyos_ultrabert import Client

    # Warmup happens automatically
    client = Client()

    # First call is already fast (~7-10ms)
    result = client.analyze("Mom picked up the kids!")
"""

import time
import threading
from typing import List, Dict, Any, Optional, Union, Generator
from dataclasses import dataclass, field
from collections import deque
import warnings

from .model import UltraBERT, AnalysisOutput
from . import __version__


# Unicode normalization map for consistent tokenization
# Handles smart quotes, curly apostrophes, and other common variations
_UNICODE_NORMALIZE_MAP = {
    "\u2018": "'",  # LEFT SINGLE QUOTATION MARK
    "\u2019": "'",  # RIGHT SINGLE QUOTATION MARK (curly apostrophe)
    "\u201A": "'",  # SINGLE LOW-9 QUOTATION MARK
    "\u201B": "'",  # SINGLE HIGH-REVERSED-9 QUOTATION MARK
    "\u201C": '"',  # LEFT DOUBLE QUOTATION MARK
    "\u201D": '"',  # RIGHT DOUBLE QUOTATION MARK
    "\u201E": '"',  # DOUBLE LOW-9 QUOTATION MARK
    "\u201F": '"',  # DOUBLE HIGH-REVERSED-9 QUOTATION MARK
    "\u2032": "'",  # PRIME
    "\u2033": '"',  # DOUBLE PRIME
    "\u2014": "-",  # EM DASH
    "\u2013": "-",  # EN DASH
    "\u00A0": " ",  # NON-BREAKING SPACE
    "\u2026": "...",  # HORIZONTAL ELLIPSIS
}

_NORMALIZE_TABLE = str.maketrans(_UNICODE_NORMALIZE_MAP)

# Safety-critical contraction expansions
# Expanding these helps the model recognize harmful intent more clearly
import re as _re

_SAFETY_PATTERNS = [
    # Harmful intent patterns - MUST be expanded for accurate detection
    (_re.compile(r"\bI'm\s+going\s+to\s+hurt\b", _re.IGNORECASE), "I am going to hurt"),
    (_re.compile(r"\bI'm\s+going\s+to\s+kill\b", _re.IGNORECASE), "I am going to kill"),
    (_re.compile(r"\bI'm\s+going\s+to\s+harm\b", _re.IGNORECASE), "I am going to harm"),
    (_re.compile(r"\bI'm\s+going\s+to\s+end\b", _re.IGNORECASE), "I am going to end"),
    # Self-harm patterns
    (_re.compile(r"\bI'm\s+going\s+to\s+hurt\s+myself\b", _re.IGNORECASE), "I am going to hurt myself"),
    (_re.compile(r"\bI'm\s+going\s+to\s+cut\b", _re.IGNORECASE), "I am going to cut"),
    # Colloquial forms
    (_re.compile(r"\bgonna\s+hurt\b", _re.IGNORECASE), "going to hurt"),
    (_re.compile(r"\bgonna\s+kill\b", _re.IGNORECASE), "going to kill"),
]


def _normalize_text(text: str) -> str:
    """Normalize text for consistent model behavior.

    Handles:
    1. Smart/curly quotes -> straight quotes
    2. Safety-critical contractions -> expanded forms

    This is critical for safety-sensitive text where variations
    could cause different model behavior.
    """
    # First: normalize Unicode
    text = text.translate(_NORMALIZE_TABLE)

    # Second: expand safety-critical contractions
    for pattern, replacement in _SAFETY_PATTERNS:
        text = pattern.sub(replacement, text)

    return text


@dataclass
class LatencyStats:
    """Track inference latency statistics."""

    window_size: int = 100
    _latencies: deque = field(default_factory=lambda: deque(maxlen=100))
    total_calls: int = 0
    total_time_ms: float = 0.0

    def record(self, latency_ms: float):
        """Record a latency measurement."""
        self._latencies.append(latency_ms)
        self.total_calls += 1
        self.total_time_ms += latency_ms

    @property
    def avg(self) -> float:
        """Average latency over recent window."""
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    @property
    def min(self) -> float:
        """Minimum latency in window."""
        return min(self._latencies) if self._latencies else 0.0

    @property
    def max(self) -> float:
        """Maximum latency in window."""
        return max(self._latencies) if self._latencies else 0.0

    @property
    def lifetime_avg(self) -> float:
        """Average latency over all calls."""
        if self.total_calls == 0:
            return 0.0
        return self.total_time_ms / self.total_calls

    def summary(self) -> Dict[str, float]:
        """Get latency summary."""
        return {
            "avg_ms": round(self.avg, 2),
            "min_ms": round(self.min, 2),
            "max_ms": round(self.max, 2),
            "lifetime_avg_ms": round(self.lifetime_avg, 2),
            "total_calls": self.total_calls,
        }


class Client:
    """
    Production-ready FamilyOS UltraBERT client.

    Handles model loading, warmup, and provides a clean API for inference.
    The model is warmed up automatically so the first user call is fast.

    Args:
        backend: "auto", "pytorch", or "onnx". Default auto-selects best.
        warmup: If True (default), warm up model on init for consistent latency.
        warmup_rounds: Number of warmup inferences to run. Default 3.
        lazy_load: If True, defer loading until first call. Default False.
        verbose: If True, print loading/warmup info. Default False.
        load_decoder: If True, load decoder for counterfactual generation. Default False.

    Example:
        >>> client = Client()  # Loads and warms up automatically
        >>> result = client.analyze("Mom picked up the kids!")
        >>> print(result.sentiment)  # "very_positive"
        >>> print(result.safety)     # "GREEN"
    """

    VERSION = "3.0.0"

    def __init__(
        self,
        backend: str = "auto",
        device: str = "auto",
        warmup: bool = True,
        warmup_rounds: int = 3,
        lazy_load: bool = False,
        verbose: bool = False,
        load_decoder: bool = False,
    ):
        self._backend_preference = backend
        self._device_preference = device
        self._warmup_enabled = warmup
        self._warmup_rounds = warmup_rounds
        self._verbose = verbose
        self._load_decoder = load_decoder
        self._model: Optional[UltraBERT] = None
        self._decoder_session: Optional[Any] = None
        self._is_ready = False
        self._lock = threading.Lock()
        self._stats = LatencyStats()

        if not lazy_load:
            self._ensure_ready()

    def _log(self, message: str):
        """Print message if verbose mode enabled."""
        if self._verbose:
            print(f"[UltraBERT] {message}")

    def _ensure_ready(self):
        """Ensure model is loaded and warmed up."""
        if self._is_ready:
            return

        with self._lock:
            if self._is_ready:
                return

            self._log("Loading model...")
            load_start = time.perf_counter()

            self._model = UltraBERT.load(
                backend=self._backend_preference,
                device=self._device_preference,
            )

            load_time = (time.perf_counter() - load_start) * 1000
            self._log(f"Model loaded in {load_time:.0f}ms (backend: {self._model.backend})")

            if self._warmup_enabled:
                self._warmup()

            self._is_ready = True

    def _warmup(self):
        """Warm up the model with sample inferences."""
        self._log(f"Warming up ({self._warmup_rounds} rounds)...")

        warmup_texts = [
            "Mom picked up the kids from school today.",
            "I love my family so much!",
            "Dad made breakfast this morning.",
        ]

        for i in range(self._warmup_rounds):
            text = warmup_texts[i % len(warmup_texts)]
            start = time.perf_counter()
            self._model.analyze(text)
            elapsed = (time.perf_counter() - start) * 1000
            self._log(f"  Warmup {i+1}: {elapsed:.1f}ms")

        self._log("Warmup complete - ready for fast inference!")

    @property
    def is_ready(self) -> bool:
        """Check if client is ready for inference."""
        return self._is_ready

    @property
    def backend(self) -> str:
        """Get the active backend ("pytorch" or "onnx")."""
        self._ensure_ready()
        return self._model.backend

    @property
    def device(self) -> str:
        """Get the active device string when available.

        Returns:
            A string such as "cpu" or "cuda" when available; otherwise
            "unknown".
        """
        self._ensure_ready()
        if self._model is None:
            return "unknown"
        return str(getattr(self._model, "device", "unknown"))

    @property
    def capabilities(self) -> List[str]:
        """List of available capabilities."""
        self._ensure_ready()
        return list(self._model.capabilities)

    @property
    def stats(self) -> Dict[str, float]:
        """Get latency statistics."""
        return self._stats.summary()

    def analyze(
        self,
        text: str,
        capabilities: Optional[List[str]] = None,
    ) -> "ClientResult":
        """
        Analyze text with all or selected capabilities.

        Args:
            text: The text to analyze.
            capabilities: Optional list of specific capabilities to run.
                         If None, runs all capabilities.

        Returns:
            ClientResult with easy access to all predictions.

        Example:
            >>> result = client.analyze("I love my mom!")
            >>> result.sentiment      # "very_positive"
            >>> result.safety         # "GREEN"
            >>> result.emotions       # ["joy", "love"]
            >>> result.latency_ms     # 7.5
        """
        self._ensure_ready()

        # Normalize text to handle smart quotes and Unicode variations
        # Critical for safety-sensitive detection
        normalized_text = _normalize_text(text)

        start = time.perf_counter()
        raw_result = self._model.analyze(normalized_text, capabilities=capabilities)
        latency = (time.perf_counter() - start) * 1000

        self._stats.record(latency)

        return ClientResult(raw_result, latency)

    def analyze_batch(
        self,
        texts: List[str],
        capabilities: Optional[List[str]] = None,
    ) -> List["ClientResult"]:
        """
        Analyze multiple texts.

        Args:
            texts: List of texts to analyze.
            capabilities: Optional capabilities to run.

        Returns:
            List of ClientResult objects.
        """
        return [self.analyze(text, capabilities) for text in texts]

    def get_sentiment(self, text: str) -> str:
        """Quick sentiment analysis. Returns: very_negative/negative/neutral/positive/very_positive."""
        result = self.analyze(text, capabilities=["sentiment"])
        return result.sentiment

    def get_emotions(self, text: str) -> List[str]:
        """Quick emotion detection. Returns list of detected emotions."""
        result = self.analyze(text, capabilities=["emotions"])
        return result.emotions

    def get_safety(self, text: str) -> str:
        """Quick safety check. Returns: GREEN/AMBER/RED/CRISIS."""
        result = self.analyze(text, capabilities=["safety_familyos"])
        return result.safety

    def get_embedding(self, text: str) -> List[float]:
        """Get 768-dimensional embedding vector."""
        result = self.analyze(text, capabilities=["embedding"])
        return result.embedding

    def is_safe(self, text: str) -> bool:
        """Check if text is safe (GREEN). Returns True/False."""
        return self.get_safety(text) == "GREEN"

    def is_crisis(self, text: str) -> bool:
        """Check if text indicates crisis. Returns True/False."""
        return self.get_safety(text) == "CRISIS"

    def get_intent(self, text: str) -> str:
        """Quick intent classification. Returns intent label."""
        result = self.analyze(text, capabilities=["intent"])
        return result.intent

    def get_ingress(self, text: str) -> str:
        """Quick routing category. Returns ingress label."""
        result = self.analyze(text, capabilities=["ingress"])
        return result.ingress

    def get_entities(self, text: str) -> List[Dict]:
        """Quick family entity extraction. Returns list of entity dicts."""
        result = self.analyze(text, capabilities=["ner_family"])
        return result.entities

    def get_temporal(self, text: str) -> List[Dict]:
        """Quick temporal expression extraction. Returns list of temporal dicts."""
        result = self.analyze(text, capabilities=["temporal"])
        return result.temporal

    def get_all_entities(self, text: str) -> Dict[str, List[Dict]]:
        """Get both family and general entities. Returns dict with 'family' and 'general' keys."""
        result = self.analyze(text, capabilities=["ner_family", "ner_general"])
        return {
            "family": result.entities,
            "general": result.general_entities,
        }

    def needs_attention(self, text: str) -> bool:
        """Check if text needs attention (AMBER, RED, or CRISIS). Returns True/False."""
        safety = self.get_safety(text)
        return safety in ("AMBER", "RED", "CRISIS")

    def is_positive(self, text: str) -> bool:
        """Check if sentiment is positive or very_positive. Returns True/False."""
        sentiment = self.get_sentiment(text)
        return sentiment in ("positive", "very_positive")

    def is_negative(self, text: str) -> bool:
        """Check if sentiment is negative or very_negative. Returns True/False."""
        sentiment = self.get_sentiment(text)
        return sentiment in ("negative", "very_negative")

    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Cosine similarity score (0.0 to 1.0).
        """
        import math

        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)

        dot = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = math.sqrt(sum(a * a for a in emb1))
        norm2 = math.sqrt(sum(b * b for b in emb2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed.

        Returns:
            List of 768-dimensional embedding vectors.
        """
        return [self.get_embedding(text) for text in texts]

    def find_similar(
        self,
        query: str,
        corpus: List[str],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find most similar texts in corpus to query.

        Args:
            query: Query text to find similar texts for.
            corpus: List of candidate texts.
            top_k: Number of top results to return.

        Returns:
            List of dicts with 'text', 'similarity', and 'index' keys.
        """
        query_emb = self.get_embedding(query)
        corpus_embs = self.embed_batch(corpus)

        import math

        def cosine(emb1, emb2):
            dot = sum(a * b for a, b in zip(emb1, emb2))
            norm1 = math.sqrt(sum(a * a for a in emb1))
            norm2 = math.sqrt(sum(b * b for b in emb2))
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot / (norm1 * norm2)

        results = []
        for i, (text, emb) in enumerate(zip(corpus, corpus_embs)):
            sim = cosine(query_emb, emb)
            results.append({"text": text, "similarity": round(sim, 4), "index": i})

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def classify_batch(
        self,
        texts: List[str],
        capability: str,
    ) -> List[Any]:
        """
        Classify multiple texts with a single capability.

        Args:
            texts: List of texts to classify.
            capability: Capability to run (e.g., "sentiment", "safety_familyos").

        Returns:
            List of predictions.
        """
        results = []
        for text in texts:
            result = self.analyze(text, capabilities=[capability])
            if capability == "sentiment":
                results.append(result.sentiment)
            elif capability == "emotions":
                results.append(result.emotions)
            elif capability == "safety_familyos":
                results.append(result.safety)
            elif capability == "intent":
                results.append(result.intent)
            elif capability == "ingress":
                results.append(result.ingress)
            elif capability == "embedding":
                results.append(result.embedding)
            else:
                results.append(result.to_dict())
        return results

    def stream_analyze(self, texts: List[str]) -> "Generator[ClientResult, None, None]":
        """
        Generator for memory-efficient batch analysis.

        Args:
            texts: List of texts to analyze.

        Yields:
            ClientResult for each text.
        """
        for text in texts:
            yield self.analyze(text)

    def export_embeddings(
        self,
        texts: List[str],
        path: str,
        format: str = "jsonl",
    ) -> int:
        """
        Export embeddings to file.

        Args:
            texts: List of texts to embed.
            path: Output file path.
            format: "jsonl" or "csv".

        Returns:
            Number of embeddings exported.
        """
        import json

        embeddings = self.embed_batch(texts)

        if format == "jsonl":
            with open(path, "w", encoding="utf-8") as f:
                for text, emb in zip(texts, embeddings):
                    f.write(json.dumps({"text": text, "embedding": emb}) + "\n")
        elif format == "csv":
            with open(path, "w", encoding="utf-8") as f:
                # Header
                f.write("text," + ",".join([f"dim_{i}" for i in range(768)]) + "\n")
                for text, emb in zip(texts, embeddings):
                    escaped_text = text.replace('"', '""')
                    f.write(f'"{escaped_text}",' + ",".join(map(str, emb)) + "\n")
        else:
            raise ValueError(f"Unknown format: {format}. Use 'jsonl' or 'csv'.")

        return len(embeddings)

    def health_check(self) -> Dict[str, Any]:
        """
        Run a health check and return status.

        Returns:
            Dict with health status, latency, and model info.
        """
        self._ensure_ready()

        start = time.perf_counter()
        result = self._model.analyze("Health check test.")
        latency = (time.perf_counter() - start) * 1000

        return {
            "status": "healthy",
            "version": self.VERSION,
            "package_version": __version__,
            "backend": self.backend,
            "capabilities": len(self.capabilities),
            "latency_ms": round(latency, 2),
            "stats": self.stats,
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get detailed latency statistics.

        Returns:
            Dict with total_calls, avg_latency_ms, min_latency_ms, max_latency_ms,
            p50_latency_ms, p95_latency_ms, p99_latency_ms.
        """
        latencies = list(self._stats._latencies)
        if not latencies:
            return {
                "total_calls": 0,
                "avg_latency_ms": 0.0,
                "min_latency_ms": 0.0,
                "max_latency_ms": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
            }

        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        def percentile(p: float) -> float:
            k = (n - 1) * p
            f = int(k)
            c = f + 1 if f + 1 < n else f
            return sorted_latencies[f] + (k - f) * (sorted_latencies[c] - sorted_latencies[f])

        return {
            "total_calls": self._stats.total_calls,
            "avg_latency_ms": round(self._stats.avg, 2),
            "min_latency_ms": round(self._stats.min, 2),
            "max_latency_ms": round(self._stats.max, 2),
            "p50_latency_ms": round(percentile(0.50), 2),
            "p95_latency_ms": round(percentile(0.95), 2),
            "p99_latency_ms": round(percentile(0.99), 2),
        }

    def reset_stats(self) -> None:
        """
        Reset latency statistics.
        """
        self._stats = LatencyStats()

    # =========================================================================
    # v3: Decoder Methods for Counterfactual Generation
    # =========================================================================

    def create_decoder_session(
        self,
        version: str = "v3",
        quantization: str = "int8",
        device: str = "auto",
    ) -> "DecoderSession":
        """
        Create a decoder session for batch counterfactual generation.

        The decoder is loaded when entering the context and automatically
        unloaded when exiting, freeing memory. This is ideal for R5 dream
        exploration phases in P03 consolidation.

        Args:
            version: Decoder version (default: "v3")
            quantization: Weight format - "fp32", "fp16", or "int8"
            device: Backend - "auto", "npu", "cuda", or "cpu"

        Returns:
            DecoderSession context manager

        Example:
            >>> with client.create_decoder_session() as decoder:
            ...     for text in texts:
            ...         encoder_output = client.encode(text)
            ...         suggestion = decoder.generate(encoder_output)
            >>> # Decoder automatically unloaded, memory freed
        """
        from .decoder_session import DecoderSession
        return DecoderSession(
            version=version,
            quantization=quantization,
            device=device,
        )

    def encode(self, text: str) -> "np.ndarray":
        """
        Get encoder hidden states for a text.

        This returns the raw encoder output that can be passed to
        DecoderSession.generate() for counterfactual generation.

        Note: This method requires the PyTorch backend to access full
        encoder hidden states. ONNX backend will raise an error.

        Args:
            text: Text to encode

        Returns:
            Encoder hidden states as numpy array.
            Shape: (1, seq_len, 768) - full sequence hidden states

        Example:
            >>> encoder_output = client.encode("I hate this situation")
            >>> with client.create_decoder_session() as decoder:
            ...     suggestion = decoder.generate(encoder_output)
        """
        import numpy as np

        self._ensure_ready()

        # Access the underlying engine to get full hidden states
        engine = self._model._engine

        # Check if we have PyTorch backend with _encode method
        if hasattr(engine, "_encode"):
            # PyTorch backend - get full sequence hidden states
            hidden_states, attention_mask, tokens, from_cache = engine._encode(text)
            # Convert to numpy: (batch=1, seq_len, 768)
            return hidden_states.cpu().numpy().astype(np.float32)
        else:
            # ONNX backend - cannot get hidden states directly
            # ONNX models output final logits, not intermediate hidden states
            raise RuntimeError(
                "encode() requires PyTorch backend for full hidden states. "
                "ONNX backend only outputs final logits per capability. "
                "Initialize with: Client(backend='pytorch') or use PyTorch model."
            )

    def suggest_alternative(
        self,
        text: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
    ) -> str:
        """
        Generate a counterfactual suggestion for the given text.

        This is a convenience method that loads the decoder temporarily,
        generates a suggestion, and unloads the decoder. For processing
        multiple texts, use create_decoder_session() instead.

        Args:
            text: Text to generate alternative for
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)

        Returns:
            Generated counterfactual text

        Example:
            >>> suggestion = client.suggest_alternative("I hate this")
            "I'm not satisfied with this"
        """
        encoder_output = self.encode(text)

        with self.create_decoder_session() as decoder:
            return decoder.generate(
                encoder_output,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

    def suggest_alternative_structured(
        self,
        text: str,
        max_new_tokens: int = 128,
    ) -> Dict[str, Any]:
        """
        Generate counterfactual with structured output including procedural insights.

        This extracts actionable insights from the generated counterfactual,
        useful for P03 memory consolidation.

        Args:
            text: Text to generate alternative for
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dictionary with:
                - "text": Cleaned generated text
                - "raw": Raw generated text
                - "generation_time_ms": Generation time
                - "procedural_insight": Extracted insight dict

        Example:
            >>> result = client.suggest_alternative_structured("I felt overwhelmed")
            >>> print(result["procedural_insight"])
            {"trigger": "feeling overwhelmed", "action": "schedule break", ...}
        """
        encoder_output = self.encode(text)

        with self.create_decoder_session() as decoder:
            return decoder.generate_structured(
                encoder_output,
                max_new_tokens=max_new_tokens,
            )

    def __repr__(self) -> str:
        status = "ready" if self._is_ready else "not loaded"
        backend = self._model.backend if self._model else "N/A"
        return f"<UltraBERT Client v{self.VERSION} ({status}, backend={backend})>"


class ClientResult:
    """
    Clean result wrapper with easy attribute access.

    Attributes:
        text: Original input text
        sentiment: Predicted sentiment (very_negative to very_positive)
        sentiment_confidence: Confidence score for sentiment
        emotions: List of detected emotions
        safety: Safety band (GREEN/AMBER/RED/CRISIS)
        safety_confidence: Confidence for safety prediction
        entities: List of detected family entities
        temporal: List of temporal expressions
        intent: Detected user intent
        ingress: Message routing category
        embedding: 768-dim embedding vector
        latency_ms: Inference time in milliseconds
    """

    def __init__(self, raw: AnalysisOutput, latency_ms: float):
        self._raw = raw
        self._latency_ms = latency_ms
        self._caps = raw.capabilities if hasattr(raw, 'capabilities') else {}

    @property
    def text(self) -> str:
        """Original input text."""
        return self._raw.text

    @property
    def latency_ms(self) -> float:
        """Inference latency in milliseconds."""
        return round(self._latency_ms, 2)

    # Sentiment
    @property
    def sentiment(self) -> str:
        """Sentiment prediction."""
        return self._caps.get("sentiment", {}).get("prediction", "unknown")

    @property
    def sentiment_confidence(self) -> float:
        """Sentiment confidence score."""
        return self._caps.get("sentiment", {}).get("confidence", 0.0)

    @property
    def sentiment_scores(self) -> Dict[str, float]:
        """All sentiment class scores."""
        return self._caps.get("sentiment", {}).get("scores", {})

    # Emotions
    @property
    def emotions(self) -> List[str]:
        """Detected emotions."""
        return self._caps.get("emotions", {}).get("predictions", [])

    @property
    def emotion_scores(self) -> Dict[str, float]:
        """All emotion scores."""
        return self._caps.get("emotions", {}).get("scores", {})

    # Safety
    @property
    def safety(self) -> str:
        """Safety band (GREEN/AMBER/RED/CRISIS)."""
        return self._caps.get("safety_familyos", {}).get("band", "unknown")

    @property
    def safety_confidence(self) -> float:
        """Safety prediction confidence."""
        return self._caps.get("safety_familyos", {}).get("confidence", 0.0)

    @property
    def safety_scores(self) -> Dict[str, float]:
        """All safety band probabilities."""
        return self._caps.get("safety_familyos", {}).get("probabilities", {})

    @property
    def is_safe(self) -> bool:
        """True if safety is GREEN."""
        return self.safety == "GREEN"

    @property
    def is_crisis(self) -> bool:
        """True if safety is CRISIS."""
        return self.safety == "CRISIS"

    # Entities
    @property
    def entities(self) -> List[Dict]:
        """Family entities (KINSHIP, etc.)."""
        return self._caps.get("ner_family", {}).get("entities", [])

    @property
    def general_entities(self) -> List[Dict]:
        """General NER entities."""
        return self._caps.get("ner_general", {}).get("entities", [])

    # Temporal
    @property
    def temporal(self) -> List[Dict]:
        """Temporal expressions."""
        return self._caps.get("temporal", {}).get("entities", [])

    # Intent
    @property
    def intent(self) -> str:
        """User intent."""
        return self._caps.get("intent", {}).get("prediction", "unknown")

    @property
    def intent_confidence(self) -> float:
        """Intent confidence."""
        return self._caps.get("intent", {}).get("confidence", 0.0)

    # Ingress
    @property
    def ingress(self) -> str:
        """Message routing category."""
        return self._caps.get("ingress", {}).get("prediction", "unknown")

    # Relation
    @property
    def relations(self) -> List[str]:
        """Detected relationship types."""
        return self._caps.get("relation", {}).get("predictions", [])

    # NLI
    @property
    def nli(self) -> str:
        """NLI prediction (entailment/neutral/contradiction)."""
        return self._caps.get("nli", {}).get("prediction", "unknown")

    # Embedding
    @property
    def embedding(self) -> List[float]:
        """768-dimensional embedding vector."""
        return self._caps.get("embedding", {}).get("embedding", [])

    @property
    def embedding_dim(self) -> int:
        """Embedding dimensionality."""
        return len(self.embedding)

    # Additional convenience properties
    @property
    def needs_attention(self) -> bool:
        """True if safety is not GREEN (needs attention)."""
        return self.safety in ("AMBER", "RED", "CRISIS")

    @property
    def top_emotion(self) -> Optional[str]:
        """Highest confidence emotion, or None if no emotions."""
        scores = self.emotion_scores
        if not scores:
            return self.emotions[0] if self.emotions else None
        return max(scores, key=scores.get) if scores else None

    @property
    def sentiment_direction(self) -> str:
        """Simplified sentiment: 'positive', 'negative', or 'neutral'."""
        s = self.sentiment
        if s in ("positive", "very_positive"):
            return "positive"
        elif s in ("negative", "very_negative"):
            return "negative"
        else:
            return "neutral"

    @property
    def has_entities(self) -> bool:
        """True if any family entities were found."""
        return len(self.entities) > 0

    @property
    def entity_texts(self) -> List[str]:
        """Just the text spans of detected entities."""
        return [e.get("text", e.get("entity", "")) for e in self.entities]

    def to_json(self) -> str:
        """Convert to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2)

    @property
    def summary(self) -> str:
        """One-line summary of the result."""
        parts = [f"safety={self.safety}"]
        if self.sentiment != "unknown":
            parts.append(f"sentiment={self.sentiment_direction}")
        if self.emotions:
            parts.append(f"emotions={self.emotions[:2]}")
        if self.entities:
            parts.append(f"entities={len(self.entities)}")
        return " | ".join(parts)

    # Utilities
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "sentiment": self.sentiment,
            "sentiment_confidence": self.sentiment_confidence,
            "emotions": self.emotions,
            "safety": self.safety,
            "safety_confidence": self.safety_confidence,
            "entities": self.entities,
            "temporal": self.temporal,
            "intent": self.intent,
            "ingress": self.ingress,
            "relations": self.relations,
            "latency_ms": self.latency_ms,
        }

    def __repr__(self) -> str:
        return (
            f"<ClientResult "
            f"sentiment={self.sentiment} "
            f"safety={self.safety} "
            f"emotions={self.emotions[:3]}{'...' if len(self.emotions) > 3 else ''} "
            f"latency={self.latency_ms}ms>"
        )


# Convenience function for quick one-off analysis
def analyze(text: str, **kwargs) -> ClientResult:
    """
    Quick one-off analysis (creates temporary client).

    For repeated use, create a Client instance instead.

    Args:
        text: Text to analyze.
        **kwargs: Passed to Client constructor.

    Returns:
        ClientResult with predictions.
    """
    client = Client(**kwargs)
    return client.analyze(text)
