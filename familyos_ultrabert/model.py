"""
FamilyOS NLP - Unified Model Loader

Simple API for loading and using the FamilyOS NLP model.
Automatically selects the best backend (PyTorch or ONNX) based on
available hardware and dependencies.

Usage:
    >>> from familyos_nlp import FamilyOSModel
    >>> model = FamilyOSModel.load()
    >>> results = model.analyze("Mom picked up Panda from school")
    >>> print(results["sentiment"])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

logger = logging.getLogger(__name__)

# Package root directory
PACKAGE_DIR = Path(__file__).parent

# Default weight paths
DEFAULT_PYTORCH_PATH = PACKAGE_DIR / "weights" / "pytorch"
DEFAULT_ONNX_PATH = PACKAGE_DIR / "weights" / "onnx"


@dataclass
class AnalysisOutput:
    """
    Simplified output from analysis.

    Attributes:
        text: Input text that was analyzed
        capabilities: Dict of capability name to output
        latency_ms: Total inference latency in milliseconds
        backend: Which backend was used ("pytorch" or "onnx")
    """

    text: str
    capabilities: Dict[str, Dict[str, Any]]
    latency_ms: float
    backend: str

    def __getitem__(self, key: str) -> Dict[str, Any]:
        """Allow dict-like access: result["sentiment"]"""
        return self.capabilities[key]

    def __contains__(self, key: str) -> bool:
        return key in self.capabilities

    def keys(self):
        return self.capabilities.keys()

    def items(self):
        return self.capabilities.items()

    def values(self):
        return self.capabilities.values()


class FamilyOSModel:
    """
    Unified interface for FamilyOS NLP inference.

    Automatically selects the best backend:
    - PyTorch (GPU): Best for multi-capability inference, GPU required
    - ONNX (CPU/GPU): Best for single-capability or CPU-only deployment

    Example:
        >>> model = FamilyOSModel.load()
        >>> result = model.analyze("I love my family!")
        >>> print(result["sentiment"])
        {'prediction': 'very_positive', 'confidence': 0.89, ...}

        >>> result = model.analyze(
        ...     "Mom picked up Panda from school",
        ...     capabilities=["ner_family", "sentiment", "safety_familyos"]
        ... )
        >>> for cap, output in result.items():
        ...     print(f"{cap}: {output}")
    """

    def __init__(
        self,
        engine: Any,
        backend: str,
        capabilities: List[str],
    ):
        """
        Initialize model wrapper.

        Use FamilyOSModel.load() instead of calling this directly.
        """
        self._engine = engine
        self._backend = backend
        self._capabilities = capabilities

    @classmethod
    def load(
        cls,
        model_path: Optional[str] = None,
        backend: Literal["auto", "pytorch", "onnx"] = "auto",
        device: Literal["auto", "cpu", "cuda"] = "auto",
        enable_cache: bool = True,
        cache_size: int = 1000,
    ) -> "FamilyOSModel":
        """
        Load FamilyOS NLP model.

        Args:
            model_path: Path to model directory. If None, uses bundled weights.
            backend: "auto" (default), "pytorch", or "onnx"
            device: "auto" (default), "cpu", or "cuda"
            enable_cache: Enable encoder caching (PyTorch only)
            cache_size: Max cached encodings

        Returns:
            FamilyOSModel instance

        Examples:
            # Auto-detect best backend
            >>> model = FamilyOSModel.load()

            # Force PyTorch on GPU
            >>> model = FamilyOSModel.load(backend="pytorch", device="cuda")

            # Force ONNX on CPU
            >>> model = FamilyOSModel.load(backend="onnx", device="cpu")

            # Use custom model path
            >>> model = FamilyOSModel.load(model_path="/path/to/model")
        """
        # Determine backend
        use_pytorch = False
        use_onnx = False

        if backend == "auto":
            # Try PyTorch first if GPU available
            try:
                import torch

                if device == "auto":
                    has_gpu = torch.cuda.is_available()
                else:
                    has_gpu = device == "cuda"

                if has_gpu:
                    use_pytorch = True
                else:
                    # Check if ONNX is available for CPU
                    try:
                        import onnxruntime

                        use_onnx = True
                    except ImportError:
                        use_pytorch = True  # Fallback to PyTorch on CPU
            except ImportError:
                # No PyTorch, must use ONNX
                try:
                    import onnxruntime

                    use_onnx = True
                except ImportError:
                    raise ImportError(
                        "Neither PyTorch nor ONNX Runtime is installed. "
                        "Install with: pip install torch  OR  pip install onnxruntime"
                    )
        elif backend == "pytorch":
            use_pytorch = True
        elif backend == "onnx":
            use_onnx = True
        else:
            raise ValueError(f"Invalid backend: {backend}")

        # Load with selected backend
        if use_pytorch:
            return cls._load_pytorch(model_path, device, enable_cache, cache_size)
        else:
            return cls._load_onnx(model_path, device)

    @classmethod
    def _load_pytorch(
        cls,
        model_path: Optional[str],
        device: str,
        enable_cache: bool,
        cache_size: int,
    ) -> "FamilyOSModel":
        """Load PyTorch backend."""
        from .pytorch_inference import PyTorchInferenceEngine

        # Resolve path
        if model_path is None:
            if DEFAULT_PYTORCH_PATH.exists():
                model_path = str(DEFAULT_PYTORCH_PATH)
            else:
                raise FileNotFoundError(
                    f"No bundled PyTorch weights found at {DEFAULT_PYTORCH_PATH}. "
                    "Please provide model_path."
                )

        engine = PyTorchInferenceEngine.load(
            model_path=model_path,
            device=device,
            enable_cache=enable_cache,
            cache_size=cache_size,
        )

        logger.info(f"Loaded PyTorch backend: {len(engine.capabilities)} capabilities")

        return cls(engine=engine, backend="pytorch", capabilities=engine.capabilities)

    @classmethod
    def _load_onnx(cls, model_path: Optional[str], device: str) -> "FamilyOSModel":
        """Load ONNX backend."""
        from .onnx_inference import ONNXInferenceEngine

        # Resolve path
        if model_path is None:
            if DEFAULT_ONNX_PATH.exists():
                model_path = str(DEFAULT_ONNX_PATH)
            else:
                raise FileNotFoundError(
                    f"No bundled ONNX weights found at {DEFAULT_ONNX_PATH}. "
                    "Please provide model_path."
                )

        actual_device = device if device != "auto" else "cpu"

        engine = ONNXInferenceEngine.load(
            model_path=model_path,
            device=actual_device,
            use_quantized=True,
        )

        logger.info(f"Loaded ONNX backend: {len(engine.capabilities)} capabilities")

        return cls(engine=engine, backend="onnx", capabilities=engine.capabilities)

    @property
    def capabilities(self) -> List[str]:
        """List of available capabilities."""
        return self._capabilities

    @property
    def backend(self) -> str:
        """Current backend: "pytorch" or "onnx"."""
        return self._backend

    def analyze(
        self,
        text: str,
        capabilities: Optional[List[str]] = None,
    ) -> AnalysisOutput:
        """
        Analyze text with one or more capabilities.

        Args:
            text: Input text to analyze
            capabilities: Specific capabilities to run. If None, runs all.

        Returns:
            AnalysisOutput with results for each capability

        Examples:
            # Run all capabilities
            >>> result = model.analyze("Hello world")

            # Run specific capabilities
            >>> result = model.analyze(
            ...     "Mom picked up Panda from school",
            ...     capabilities=["sentiment", "ner_family", "safety_familyos"]
            ... )

            # Access results
            >>> print(result["sentiment"]["prediction"])
            'positive'

            >>> print(result["ner_family"]["entities"])
            [{'text': 'Mom', 'label': 'KINSHIP'}, {'text': 'Panda', 'label': 'NICKNAME'}]
        """
        # Run inference
        if self._backend == "pytorch":
            result = self._engine.analyze(text, capabilities)
        else:
            result = self._engine.analyze(text, capabilities)

        # Simplify output
        outputs = {cap: res.output for cap, res in result.results.items()}

        return AnalysisOutput(
            text=text,
            capabilities=outputs,
            latency_ms=result.total_latency_ms,
            backend=self._backend,
        )

    def analyze_batch(
        self,
        texts: List[str],
        capabilities: Optional[List[str]] = None,
    ) -> List[AnalysisOutput]:
        """
        Analyze multiple texts.

        Args:
            texts: List of input texts
            capabilities: Specific capabilities to run

        Returns:
            List of AnalysisOutput, one per text
        """
        return [self.analyze(text, capabilities) for text in texts]

    def get_embedding(self, text: str) -> List[float]:
        """
        Get sentence embedding for text.

        Args:
            text: Input text

        Returns:
            768-dimensional embedding as list of floats

        Example:
            >>> embedding = model.get_embedding("Hello world")
            >>> len(embedding)
            768
        """
        result = self.analyze(text, capabilities=["embedding"])
        return result["embedding"]["embedding"]

    def get_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Get sentiment analysis for text.

        Args:
            text: Input text

        Returns:
            Dict with 'prediction', 'confidence', and 'scores'

        Example:
            >>> model.get_sentiment("I love this!")
            {'prediction': 'very_positive', 'confidence': 0.92, 'scores': {...}}
        """
        result = self.analyze(text, capabilities=["sentiment"])
        return result["sentiment"]

    def get_emotions(self, text: str, threshold: float = 0.3) -> List[str]:
        """
        Get detected emotions for text.

        Args:
            text: Input text
            threshold: Minimum probability threshold

        Returns:
            List of detected emotion labels

        Example:
            >>> model.get_emotions("I'm so happy to see you!")
            ['joy', 'excitement', 'love']
        """
        result = self.analyze(text, capabilities=["emotions"])
        return result["emotions"]["predictions"]

    def get_safety_band(self, text: str) -> str:
        """
        Get safety band for text.

        Args:
            text: Input text

        Returns:
            Safety band: "GREEN", "AMBER", "RED", or "CRISIS"

        Example:
            >>> model.get_safety_band("Having a great day!")
            'GREEN'
        """
        result = self.analyze(text, capabilities=["safety_familyos"])
        return result["safety_familyos"]["band"]

    def get_entities(
        self, text: str, entity_type: Literal["family", "general"] = "family"
    ) -> List[Dict[str, Any]]:
        """
        Extract named entities from text.

        Args:
            text: Input text
            entity_type: "family" for family-specific NER, "general" for standard NER

        Returns:
            List of entity dicts with 'text' and 'label' keys

        Example:
            >>> model.get_entities("Mom picked up Panda from school")
            [{'text': 'Mom', 'label': 'KINSHIP'}, {'text': 'Panda', 'label': 'NICKNAME'}]
        """
        cap = "ner_family" if entity_type == "family" else "ner_general"
        result = self.analyze(text, capabilities=[cap])
        return result[cap]["entities"]

    def clear_cache(self) -> None:
        """Clear encoder cache (PyTorch backend only)."""
        if hasattr(self._engine, "clear_cache"):
            self._engine.clear_cache()

    def cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics (PyTorch backend only)."""
        if hasattr(self._engine, "cache_stats"):
            return self._engine.cache_stats()
        return {"enabled": False}
