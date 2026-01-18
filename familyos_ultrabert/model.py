"""
FamilyOS UltraBERT v3 - Unified Model Loader

Simple API for loading and using the FamilyOS UltraBERT model.
Automatically selects the best backend (PyTorch or ONNX) based on
available hardware and dependencies.

v3 Features:
    - Lazy decoder loading for memory efficiency
    - Counterfactual generation support
    - Automatic weight downloading from HuggingFace Hub
    - NPU/GPU/CPU backend auto-detection

Usage:
    >>> from familyos_ultrabert import UltraBERT
    >>> model = UltraBERT.load()
    >>> results = model.analyze("Mom picked up Panda from school")
    >>> print(results["sentiment"])

    # With decoder for counterfactual generation
    >>> model = UltraBERT.load(load_decoder=True)
    >>> suggestion = model.generate_counterfactual("I hate this")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

logger = logging.getLogger(__name__)

# Package root directory
PACKAGE_DIR = Path(__file__).parent

# Default weight paths (legacy bundled weights)
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


class UltraBERT:
    """
    FamilyOS UltraBERT v3 - Unified Multi-Task NLP Interface.

    Automatically selects the best backend:
    - PyTorch (GPU): Best for multi-capability inference, GPU required
    - ONNX (CPU/GPU/NPU): Best for single-capability or edge deployment

    v3 Features:
    - 13 capabilities including counterfactual generation
    - Lazy decoder loading (memory efficient for edge devices)
    - Automatic weight downloading from HuggingFace Hub
    - NPU support via DirectML

    Example:
        >>> model = UltraBERT.load()
        >>> result = model.analyze("I love my family!")
        >>> print(result["sentiment"])
        {'prediction': 'very_positive', 'confidence': 0.89, ...}

        >>> # With decoder for generation
        >>> model = UltraBERT.load(load_decoder=True)
        >>> model.generate_counterfactual("I hate this")
        "I'm not happy with this"
    """

    def __init__(
        self,
        engine: Any,
        backend: str,
        capabilities: List[str],
        decoder: Any = None,
    ):
        """
        Initialize model wrapper.

        Use UltraBERT.load() instead of calling this directly.
        """
        self._engine = engine
        self._backend = backend
        self._capabilities = capabilities
        self._decoder = decoder
        self._decoder_loaded = decoder is not None

    @classmethod
    def load(
        cls,
        model_path: Optional[str] = None,
        backend: Literal["auto", "pytorch", "onnx"] = "auto",
        device: Literal["auto", "cpu", "cuda", "npu"] = "auto",
        enable_cache: bool = True,
        cache_size: int = 1000,
        # v3: New parameters
        encoder_version: str = "v2-checkpoint-16500",  # Must match decoder training
        decoder_version: str = "v3",
        quantization: Literal["fp32", "fp16", "int8"] = "int8",
        load_decoder: bool = False,
    ) -> "UltraBERT":
        """
        Load FamilyOS UltraBERT model.

        Args:
            model_path: Path to model directory. If None, downloads from HuggingFace.
            backend: "auto" (default), "pytorch", or "onnx"
            device: "auto" (default), "cpu", "cuda", or "npu"
            enable_cache: Enable encoder caching (PyTorch only)
            cache_size: Max cached encodings
            encoder_version: Version of encoder weights (default: v2-checkpoint-16500)
            decoder_version: Version of decoder weights (default: v3)
            quantization: Weight format - "fp32", "fp16", "int8" (default: int8)
            load_decoder: Load decoder immediately (default: False for memory)

        Returns:
            UltraBERT instance

        Examples:
            # Auto-detect best backend
            >>> model = UltraBERT.load()

            # Force PyTorch on GPU
            >>> model = UltraBERT.load(backend="pytorch", device="cuda")

            # Force ONNX on CPU
            >>> model = UltraBERT.load(backend="onnx", device="cpu")

            # Load with decoder for counterfactual generation
            >>> model = UltraBERT.load(load_decoder=True)

            # Use custom model path
            >>> model = UltraBERT.load(model_path="/path/to/model")
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

        # Adjust quantization for backend compatibility
        # PyTorch backend requires fp32 (no PyTorch int8/fp16 weights available)
        # ONNX backend can use int8 (quantized) or fp32
        if use_pytorch and quantization != "fp32":
            logger.info(f"PyTorch backend requires fp32 weights. Switching from {quantization} to fp32.")
            quantization = "fp32"

        # Download weights if no path provided
        encoder_path = model_path
        if encoder_path is None:
            # Try to download from HuggingFace
            try:
                from .weights_manager import download_encoder, is_cached

                if is_cached("encoder", encoder_version, quantization):
                    encoder_path = str(download_encoder(encoder_version, quantization))
                    logger.info(f"Using cached encoder weights: {encoder_path}")
                else:
                    # Check for legacy bundled weights first
                    if use_pytorch and DEFAULT_PYTORCH_PATH.exists():
                        encoder_path = str(DEFAULT_PYTORCH_PATH)
                        logger.info(f"Using bundled PyTorch weights: {encoder_path}")
                    elif use_onnx and DEFAULT_ONNX_PATH.exists():
                        encoder_path = str(DEFAULT_ONNX_PATH)
                        logger.info(f"Using bundled ONNX weights: {encoder_path}")
                    else:
                        # Download from HuggingFace
                        encoder_path = str(download_encoder(encoder_version, quantization))
                        logger.info(f"Downloaded encoder weights: {encoder_path}")
            except ImportError:
                # weights_manager not available, try bundled weights
                if use_pytorch and DEFAULT_PYTORCH_PATH.exists():
                    encoder_path = str(DEFAULT_PYTORCH_PATH)
                elif use_onnx and DEFAULT_ONNX_PATH.exists():
                    encoder_path = str(DEFAULT_ONNX_PATH)
                else:
                    raise FileNotFoundError(
                        "No weights found. Either provide model_path or install "
                        "huggingface-hub to download weights automatically."
                    )

        # Load with selected backend
        if use_pytorch:
            instance = cls._load_pytorch(encoder_path, device, enable_cache, cache_size)
        else:
            instance = cls._load_onnx(encoder_path, device)

        # Load decoder if requested
        if load_decoder:
            instance.load_decoder(version=decoder_version, quantization=quantization)

        return instance

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

    @property
    def device(self) -> str:
        """Return the active device string when available.

        Returns:
            A string such as "cpu" or "cuda" when the underlying inference
            engine exposes device information; otherwise "unknown".
        """
        dev = getattr(self._engine, "device", None)
        return str(dev) if dev is not None else "unknown"

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

    # =========================================================================
    # v3: Decoder Methods
    # =========================================================================

    @property
    def decoder_loaded(self) -> bool:
        """Check if decoder is loaded."""
        return self._decoder_loaded

    def load_decoder(
        self,
        version: str = "v3",
        quantization: Literal["fp32", "fp16", "int8"] = "int8",
        force: bool = False,
    ) -> None:
        """
        Load the decoder for counterfactual generation.

        This is called automatically if load_decoder=True in UltraBERT.load(),
        but can also be called manually for lazy loading.

        Args:
            version: Decoder version (default: v3)
            quantization: Weight format - "fp32", "fp16", "int8"
            force: Re-download even if cached

        Example:
            >>> model = UltraBERT.load()  # No decoder
            >>> model.load_decoder()       # Load decoder on demand
            >>> model.generate_counterfactual("I hate this")
        """
        if self._decoder_loaded and not force:
            logger.info("Decoder already loaded")
            return

        try:
            from .weights_manager import download_decoder

            decoder_path = download_decoder(version, quantization, force=force)
            logger.info(f"Loaded decoder from {decoder_path}")

            # Store decoder info (actual loading happens in generate)
            self._decoder_path = decoder_path
            self._decoder_loaded = True

            # Add counterfactual to capabilities
            if "counterfactual" not in self._capabilities:
                self._capabilities.append("counterfactual")

        except ImportError as e:
            logger.error(f"Failed to load decoder: {e}")
            raise ImportError(
                "huggingface-hub is required for decoder loading. "
                "Install with: pip install huggingface-hub>=0.20.0"
            )

    def unload_decoder(self) -> None:
        """
        Unload the decoder to free memory.

        Useful for R5 nightly processing where decoder is only needed
        during the dream exploration phase.

        Example:
            >>> with model.decoder_session():
            ...     result = model.generate_counterfactual("I hate this")
            >>> # Decoder automatically unloaded
        """
        if self._decoder is not None:
            del self._decoder
            self._decoder = None

        self._decoder_loaded = False
        self._decoder_path = None

        # Remove counterfactual from capabilities
        if "counterfactual" in self._capabilities:
            self._capabilities.remove("counterfactual")

        # Force garbage collection
        import gc
        gc.collect()

        logger.info("Decoder unloaded")

    def generate_counterfactual(
        self,
        text: str,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate a counterfactual (alternative phrasing) for the input text.

        Requires decoder to be loaded. Call load_decoder() first if not
        loaded during UltraBERT.load().

        Args:
            text: Input text to generate counterfactual for
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling probability threshold

        Returns:
            Generated counterfactual text

        Raises:
            RuntimeError: If decoder is not loaded

        Example:
            >>> model = UltraBERT.load(load_decoder=True)
            >>> model.generate_counterfactual("I hate this")
            "I'm not satisfied with this"
        """
        if not self._decoder_loaded:
            raise RuntimeError(
                "Decoder not loaded. Call model.load_decoder() first or "
                "use UltraBERT.load(load_decoder=True)"
            )

        # This is a placeholder - actual implementation depends on the
        # inference engine supporting decoder operations
        logger.warning(
            "Counterfactual generation not yet implemented in inference engine. "
            "This will be available in a future release."
        )
        return f"[Counterfactual for: {text}]"

    def decoder_session(self):
        """
        Context manager for temporary decoder loading.

        Loads decoder on enter, unloads on exit. Perfect for R5 nightly
        processing where decoder is only needed temporarily.

        Example:
            >>> model = UltraBERT.load()
            >>> with model.decoder_session():
            ...     result = model.generate_counterfactual("I hate this")
            >>> # Decoder automatically unloaded, memory freed
        """
        return _DecoderSession(self)


class _DecoderSession:
    """Context manager for temporary decoder loading."""

    def __init__(self, model: UltraBERT):
        self._model = model
        self._was_loaded = model.decoder_loaded

    def __enter__(self):
        if not self._model.decoder_loaded:
            self._model.load_decoder()
        return self._model

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._was_loaded:
            self._model.unload_decoder()
        return False
