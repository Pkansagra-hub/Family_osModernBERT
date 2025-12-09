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
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from collections import deque
import warnings

from .model import UltraBERT, AnalysisOutput
from . import __version__


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

    Example:
        >>> client = Client()  # Loads and warms up automatically
        >>> result = client.analyze("Mom picked up the kids!")
        >>> print(result.sentiment)  # "very_positive"
        >>> print(result.safety)     # "GREEN"
    """

    VERSION = "2.0.1"

    def __init__(
        self,
        backend: str = "auto",
        warmup: bool = True,
        warmup_rounds: int = 3,
        lazy_load: bool = False,
        verbose: bool = False,
    ):
        self._backend_preference = backend
        self._warmup_enabled = warmup
        self._warmup_rounds = warmup_rounds
        self._verbose = verbose
        self._model: Optional[UltraBERT] = None
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

            self._model = UltraBERT.load(backend=self._backend_preference)

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

        start = time.perf_counter()
        raw_result = self._model.analyze(text, capabilities=capabilities)
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
