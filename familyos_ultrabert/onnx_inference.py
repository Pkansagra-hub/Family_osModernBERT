"""
FamilyOS UltraBERT v4 - ONNX Inference Backend

Lightweight inference using ONNX Runtime with:
- CPU and GPU execution providers
- Dynamic INT8 quantized models for faster CPU inference
- GlobalPointer span-based NER decoding (v4)
- No PyTorch dependency required

Note: ONNX models run each capability independently (no shared encoder),
so multi-capability inference is slower than PyTorch backend.
Use ONNX for single-capability inference or CPU-only deployment.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from .labels import CAPABILITY_TO_LABELS, CAPABILITY_TO_GP_LABELS, Capability, LabelSchema


logger = logging.getLogger(__name__)


def _load_tokenizer(model_path: Union[str, Path]):
    """Load tokenizer with fallback to tokenizer.json for packaged checkpoints."""
    model_path = Path(model_path)
    try:
        return AutoTokenizer.from_pretrained(str(model_path))
    except Exception as exc:
        tokenizer_file = model_path / "tokenizer.json"
        tokenizer_config_file = model_path / "tokenizer_config.json"
        if not tokenizer_file.exists():
            raise

        config: Dict[str, Any] = {}
        if tokenizer_config_file.exists():
            with open(tokenizer_config_file, encoding="utf-8") as f:
                config = json.load(f)

        kwargs = {
            "tokenizer_file": str(tokenizer_file),
            "model_max_length": config.get("model_max_length", 512),
            "clean_up_tokenization_spaces": config.get("clean_up_tokenization_spaces", True),
            "unk_token": config.get("unk_token"),
            "pad_token": config.get("pad_token"),
            "cls_token": config.get("cls_token"),
            "sep_token": config.get("sep_token"),
            "mask_token": config.get("mask_token"),
            "bos_token": config.get("bos_token"),
            "eos_token": config.get("eos_token"),
            "padding_side": config.get("padding_side", "right"),
            "truncation_side": config.get("truncation_side", "right"),
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        logger.warning(
            "Falling back to PreTrainedTokenizerFast for %s due to tokenizer metadata error: %s",
            model_path,
            exc,
        )
        return PreTrainedTokenizerFast(**kwargs)


# =============================================================================
# Per-Head Thresholds (optimal values from validation)
# =============================================================================

# These are logit thresholds - optimized via per-head grid search on validation
# Lower = more entities detected (higher recall), Higher = fewer entities (higher precision)
# Optimized for best model (familyos_ultrabert/weights/pytorch) with 0.1 granularity
DEFAULT_THRESHOLDS = {
    "ner_general": -1.0,  # F1=0.730 (P=0.740, R=0.720)
    "ner_family": -0.7,  # F1=0.812 (P=0.922, R=0.726)
    "temporal": -1.9,  # F1=0.639 (P=0.651, R=0.627)
    # V2 Label-Description Embedding heads (probability thresholds)
    "intent": 0.5,
    "intent_v2": 0.5,  # Multi-label intent classification
    "ingress": 0.5,
    "ingress_v2": 0.5,  # Multi-label domain classification
}


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


def _postprocess_label_description_intent(
    logits: np.ndarray,
    schema: LabelSchema,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Multi-label intent classification with K1-compliant output.

    Returns:
        Dict with:
            - primary: Highest-scoring intent (always returned)
            - all: List of intents above threshold
            - scores: Dict of intent -> probability
            - confidence: Probability of primary intent
    """
    probs = _sigmoid(logits[0])
    scores = {schema.id2label[i]: round(float(p), 4) for i, p in enumerate(probs)}

    # Primary = highest score (always returned)
    primary_idx = int(np.argmax(probs))
    primary = schema.id2label[primary_idx]

    # All = labels above threshold
    all_labels = [schema.id2label[i] for i, p in enumerate(probs) if p >= threshold]

    return {
        "primary": primary,
        "all": all_labels,
        "scores": scores,
        "confidence": round(float(probs[primary_idx]), 4),
    }


def _postprocess_label_description_ingress(
    logits: np.ndarray,
    schema: LabelSchema,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Multi-label ingress (domain) classification with K1-compliant output.

    Returns:
        Dict with:
            - primary: Highest-scoring domain (always returned)
            - confidence: Probability of primary domain
        Dict with:
            - domains: List of domains above threshold
            - scores: Dict of domain -> probability
    """
    probs = _sigmoid(logits[0])
    scores = {schema.id2label[i]: round(float(p), 4) for i, p in enumerate(probs)}

    primary_idx = int(np.argmax(probs))
    primary = schema.id2label[primary_idx]

    # Domains = all labels above threshold
    domains = [schema.id2label[i] for i, p in enumerate(probs) if p >= threshold]

    return {
        "primary": primary,
        "confidence": round(float(probs[primary_idx]), 4),
        "domains": domains,
        "scores": scores,
    }


def _postprocess_globalpointer(
    logits: np.ndarray,
    text: str,
    offset_mapping: List[tuple[int, int]],
    schema: LabelSchema,
    threshold: float = -1.0,
) -> Dict[str, Any]:
    """
    Decode GlobalPointer span scores to entity format.

    GlobalPointer outputs a (batch, num_labels, seq_len, seq_len) tensor where
    logits[b, l, i, j] represents the score for entity type l spanning from
    token i to token j (inclusive).

    Args:
        logits: (batch, num_labels, seq_len, seq_len) span scores
        text: Original input text for extracting entity strings
        offset_mapping: Token offset mapping from tokenizer
        schema: LabelSchema with id2label mapping
        threshold: Minimum logit threshold for entity detection (default -1.0)

    Returns:
        Dict with 'entities' list containing detected spans
    """
    entities = []

    # Remove batch dimension: (num_labels, seq_len, seq_len)
    span_scores = logits[0]

    num_labels = span_scores.shape[0]
    seq_len = span_scores.shape[1]

    for label_id in range(num_labels):
        # Only check upper triangle (start <= end)
        for tok_start in range(seq_len):
            for tok_end in range(tok_start, seq_len):
                logit = float(span_scores[label_id, tok_start, tok_end])
                if logit > threshold:
                    # Convert token span to character span
                    if tok_start >= len(offset_mapping) or tok_end >= len(offset_mapping):
                        continue

                    char_start, _ = offset_mapping[tok_start]
                    _, char_end = offset_mapping[tok_end]

                    # Skip special tokens (offsets are (0, 0))
                    if char_start == 0 and char_end == 0 and tok_start > 0:
                        continue

                    entity_text = text[char_start:char_end]
                    if not entity_text.strip():
                        continue

                    # Convert logit to probability for output score
                    prob = 1.0 / (1.0 + np.exp(-logit))

                    entities.append(
                        {
                            "text": entity_text,
                            "label": schema.id2label[label_id],
                            "start": char_start,
                            "end": char_end,
                            "start_token": tok_start,
                            "end_token": tok_end,
                            "score": round(float(prob), 4),
                        }
                    )

    # Sort by start position, then by score (descending)
    entities.sort(key=lambda x: (x["start"], -x["score"]))

    # Remove overlapping entities (keep highest score)
    filtered = []
    for entity in entities:
        overlap = False
        for kept in filtered:
            # Check for overlap
            if not (entity["end"] <= kept["start"] or entity["start"] >= kept["end"]):
                overlap = True
                break
        if not overlap:
            filtered.append(entity)

    return {"entities": filtered}


def postprocess(
    capability: str,
    logits: np.ndarray,
    tokens: List[str],
    text: str = "",
    offset_mapping: Optional[List[tuple[int, int]]] = None,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Post-process output based on capability type.

    Args:
        capability: The capability name
        logits: Model output logits
        tokens: Tokenized input tokens (for BIO-based NER fallback)
        text: Original input text (for GlobalPointer span extraction)
        offset_mapping: Token offset mapping (for GlobalPointer)
        threshold: Override threshold for GlobalPointer (uses DEFAULT_THRESHOLDS if None)

    Returns:
        Dict with processed predictions
    """
    cap_enum = Capability(capability)
    schema = CAPABILITY_TO_LABELS.get(cap_enum)

    if capability in ["ner_general", "ner_family", "temporal"]:
        # GlobalPointer: 4D output (batch, num_labels, seq, seq)
        if logits.ndim == 4 and offset_mapping is not None:
            # Use GlobalPointer label schema (no BIO format)
            gp_schema = CAPABILITY_TO_GP_LABELS.get(cap_enum, schema)
            # Use provided threshold or default per-head threshold
            thresh = threshold if threshold is not None else DEFAULT_THRESHOLDS.get(capability, 0.0)
            return _postprocess_globalpointer(
                logits, text, offset_mapping, gp_schema, threshold=thresh
            )
        # Fallback: BIO token classification (3D: batch, seq, num_labels)
        return _postprocess_token_classification(logits, tokens, schema)
    elif capability == "embedding":
        return _postprocess_embedding(logits)
    elif capability == "safety_familyos":
        return _postprocess_safety(logits, schema)
    elif capability in ["emotions", "safety_generic", "relation"]:
        return _postprocess_sequence_multi(logits, schema)
    elif capability in ["intent", "intent_v2"]:
        # V2 Label-Description Embedding: multi-label intent
        thresh = threshold if threshold is not None else DEFAULT_THRESHOLDS.get(capability, 0.5)
        return _postprocess_label_description_intent(logits, schema, threshold=thresh)
    elif capability in ["ingress", "ingress_v2"]:
        # V2 Label-Description Embedding: multi-label ingress
        thresh = threshold if threshold is not None else DEFAULT_THRESHOLDS.get(capability, 0.5)
        return _postprocess_label_description_ingress(logits, schema, threshold=thresh)
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

        tokenizer = _load_tokenizer(tokenizer_path)

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

        # Tokenize with offset mapping for GlobalPointer
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors="np",
            return_offsets_mapping=True,
        )
        offset_mapping = inputs.pop("offset_mapping")[0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        ort_inputs = {k: v for k, v in inputs.items()}

        # Run each capability
        results = {}
        for cap in capabilities:
            if cap not in self.sessions:
                continue

            cap_start = time.perf_counter()
            session = self.sessions[cap]
            session_input_names = {model_input.name for model_input in session.get_inputs()}
            cap_inputs = {}
            for name, value in ort_inputs.items():
                if name not in session_input_names:
                    continue
                if np.issubdtype(value.dtype, np.integer) and value.dtype != np.int64:
                    cap_inputs[name] = value.astype(np.int64)
                else:
                    cap_inputs[name] = value
            outputs = session.run(None, cap_inputs)
            logits = outputs[0]

            output = postprocess(
                cap,
                logits,
                tokens,
                text=text,
                offset_mapping=offset_mapping,
            )
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
