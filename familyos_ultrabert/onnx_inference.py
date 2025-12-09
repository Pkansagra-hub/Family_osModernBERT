"""
FamilyOS NLP - ONNX Inference Backend

Lightweight inference using ONNX Runtime with:
- CPU and GPU execution providers
- Dynamic INT8 quantized models for faster CPU inference
- No PyTorch dependency required

Note: ONNX models run each capability independently (no shared encoder),
so multi-capability inference is slower than PyTorch backend.
Use ONNX for single-capability inference or CPU-only deployment.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from transformers import AutoTokenizer

from .labels import CAPABILITY_TO_LABELS, Capability, LabelSchema

logger = logging.getLogger(__name__)


# =============================================================================
# Result Dataclasses
# =============================================================================


@dataclass
class InferenceResult:
    """Result for a single capability."""

    capability: str
    output: Dict[str, Any]
    latency_ms: float


@dataclass
class AnalysisResult:
    """Result from analysis."""

    text: str
    results: Dict[str, InferenceResult]
    total_latency_ms: float


# =============================================================================
# Post-Processing (same as PyTorch)
# =============================================================================


def _postprocess_token_classification(
    logits: np.ndarray, tokens: List[str], schema: LabelSchema
) -> Dict[str, Any]:
    """Extract entities from token classification logits."""
    pred_ids = np.argmax(logits, axis=-1)[0]
    pred_labels = [schema.id2label[int(i)] for i in pred_ids]

    entities = []
    current_entity = None
    special_tokens = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"}

    for i, (token, label) in enumerate(zip(tokens, pred_labels)):
        if token in special_tokens:
            if current_entity:
                entities.append(current_entity)
                current_entity = None
            continue

        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "text": token.replace("##", "").replace("Ġ", " ").strip(),
                "label": label[2:],
                "start_token": i,
                "end_token": i,
            }
        elif label.startswith("I-") and current_entity:
            if label[2:] == current_entity["label"]:
                current_entity["text"] += token.replace("##", "").replace("Ġ", " ")
                current_entity["end_token"] = i
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    return {"entities": entities}


def _softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _postprocess_sequence_single(logits: np.ndarray, schema: LabelSchema) -> Dict[str, Any]:
    """Single-label sequence classification."""
    probs = _softmax(logits[0])
    pred_idx = int(np.argmax(probs))
    return {
        "prediction": schema.id2label[pred_idx],
        "confidence": round(float(probs[pred_idx]), 4),
        "scores": {schema.id2label[i]: round(float(p), 4) for i, p in enumerate(probs)},
    }


def _postprocess_sequence_multi(
    logits: np.ndarray, schema: LabelSchema, threshold: float = 0.3
) -> Dict[str, Any]:
    """Multi-label sequence classification."""
    probs = _sigmoid(logits[0])
    predictions = []
    scores = {}
    for i, p in enumerate(probs):
        label = schema.id2label[i]
        scores[label] = round(float(p), 4)
        if p >= threshold:
            predictions.append(label)
    return {"predictions": predictions, "scores": scores}


def _postprocess_embedding(logits: np.ndarray) -> Dict[str, Any]:
    """Process embedding output."""
    embedding = logits[0]
    return {
        "embedding": embedding.tolist(),
        "dim": len(embedding),
        "norm": float(np.linalg.norm(embedding)),
    }


def _postprocess_safety(logits: np.ndarray, schema: LabelSchema) -> Dict[str, Any]:
    """Safety band classification."""
    probs = _softmax(logits[0])
    pred_idx = int(np.argmax(probs))
    return {
        "band": schema.id2label[pred_idx],
        "confidence": round(float(probs[pred_idx]), 4),
        "probabilities": {schema.id2label[i]: round(float(p), 4) for i, p in enumerate(probs)},
    }


def postprocess(capability: str, logits: np.ndarray, tokens: List[str]) -> Dict[str, Any]:
    """Post-process output based on capability type."""
    schema = CAPABILITY_TO_LABELS.get(Capability(capability))

    if capability in ["ner_general", "ner_family", "temporal"]:
        return _postprocess_token_classification(logits, tokens, schema)
    elif capability == "embedding":
        return _postprocess_embedding(logits)
    elif capability == "safety_familyos":
        return _postprocess_safety(logits, schema)
    elif capability in ["emotions", "safety_generic", "relation"]:
        return _postprocess_sequence_multi(logits, schema)
    else:
        return _postprocess_sequence_single(logits, schema)


# =============================================================================
# ONNX Inference Engine
# =============================================================================


class ONNXInferenceEngine:
    """
    ONNX Runtime inference engine.

    Features:
    - CPU and GPU (CUDA) execution
    - Quantized models for faster CPU inference
    - No PyTorch dependency

    Note: Each ONNX model contains its own encoder copy, so
    multi-capability inference runs N separate forward passes.
    For multi-capability workloads, prefer PyTorchInferenceEngine.

    Example:
        >>> engine = ONNXInferenceEngine.load("./weights/onnx")
        >>> result = engine.analyze("Hello world", capabilities=["sentiment"])
        >>> print(result.results["sentiment"].output)
    """

    def __init__(
        self,
        sessions: Dict[str, ort.InferenceSession],
        tokenizer: AutoTokenizer,
        capabilities: List[str],
        device: str = "cpu",
    ):
        if ort is None:
            raise ImportError("onnxruntime is required. Install with: pip install onnxruntime")

        self.sessions = sessions
        self.tokenizer = tokenizer
        self.capabilities = capabilities
        self.device = device

    @classmethod
    def load(
        cls,
        model_path: str,
        device: str = "cpu",
        use_quantized: bool = True,
    ) -> "ONNXInferenceEngine":
        """
        Load ONNX models from directory.

        Args:
            model_path: Path to ONNX models directory
            device: "cpu" or "cuda"
            use_quantized: Prefer quantized models (faster on CPU)

        Returns:
            ONNXInferenceEngine instance
        """
        if ort is None:
            raise ImportError("onnxruntime is required. Install with: pip install onnxruntime")

        model_path = Path(model_path)

        # Determine execution providers
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        # Session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Find and load ONNX models
        sessions = {}
        capabilities = []

        for onnx_file in model_path.glob("*.onnx"):
            name = onnx_file.stem

            # Skip quantized if we want base, or vice versa
            if "_quantized" in name:
                if use_quantized:
                    cap_name = name.replace("_quantized_dynamic", "").replace("_quantized", "")
                else:
                    continue
            else:
                if use_quantized:
                    # Check if quantized version exists
                    quantized = model_path / f"{name}_quantized_dynamic.onnx"
                    if quantized.exists():
                        continue  # Skip base, use quantized
                cap_name = name

            if cap_name in sessions:
                continue

            try:
                sessions[cap_name] = ort.InferenceSession(
                    str(onnx_file), sess_options, providers=providers
                )
                capabilities.append(cap_name)
                logger.debug(f"Loaded ONNX model: {onnx_file.name}")
            except Exception as e:
                logger.warning(f"Failed to load {onnx_file}: {e}")

        if not sessions:
            raise FileNotFoundError(f"No ONNX models found in {model_path}")

        # Load tokenizer (from parent pytorch directory if available)
        tokenizer_path = model_path
        if not (model_path / "tokenizer_config.json").exists():
            # Try parent directory with pytorch weights
            parent = model_path.parent
            for candidate in [parent / "pytorch", parent / "pruned-15pct", parent]:
                if (candidate / "tokenizer_config.json").exists():
                    tokenizer_path = candidate
                    break

        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

        logger.info(f"Loaded {len(sessions)} ONNX models: {capabilities}")

        return cls(
            sessions=sessions,
            tokenizer=tokenizer,
            capabilities=capabilities,
            device=device,
        )

    def analyze(
        self,
        text: str,
        capabilities: Optional[List[str]] = None,
    ) -> AnalysisResult:
        """
        Analyze text with specified capabilities.

        Args:
            text: Input text
            capabilities: List of capabilities (None = all available)

        Returns:
            AnalysisResult with outputs for each capability
        """
        start_time = time.perf_counter()

        # Resolve capabilities
        if capabilities is None:
            capabilities = self.capabilities
        else:
            capabilities = [c for c in capabilities if c in self.capabilities]

        if not capabilities:
            raise ValueError(f"No valid capabilities. Available: {self.capabilities}")

        # Tokenize
        inputs = self.tokenizer(text, truncation=True, max_length=512, return_tensors="np")
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        ort_inputs = {k: v for k, v in inputs.items()}

        # Run each capability
        results = {}
        for cap in capabilities:
            if cap not in self.sessions:
                continue

            cap_start = time.perf_counter()
            outputs = self.sessions[cap].run(None, ort_inputs)
            logits = outputs[0]

            output = postprocess(cap, logits, tokens)
            cap_ms = (time.perf_counter() - cap_start) * 1000

            results[cap] = InferenceResult(capability=cap, output=output, latency_ms=cap_ms)

        total_ms = (time.perf_counter() - start_time) * 1000

        return AnalysisResult(
            text=text,
            results=results,
            total_latency_ms=total_ms,
        )

    def analyze_batch(
        self,
        texts: List[str],
        capabilities: Optional[List[str]] = None,
    ) -> List[AnalysisResult]:
        """Analyze multiple texts."""
        return [self.analyze(text, capabilities) for text in texts]
