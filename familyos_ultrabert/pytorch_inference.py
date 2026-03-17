"""
FamilyOS UltraBERT v4 - PyTorch Inference Backend

High-performance inference using PyTorch with:
- Single encoder pass for multiple capabilities
- Parallel head execution via CUDA streams
- GlobalPointer span-based NER decoding (v4)
- Optional encoder output caching

This module loads the full multi-task model from the training codebase.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from .labels import CAPABILITY_TO_LABELS, CAPABILITY_TO_GP_LABELS, Capability, LabelSchema


logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODE = "document"


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
# Optimized for best model (best_v4_halo checkpoint) with 0.1 granularity
DEFAULT_THRESHOLDS = {
    # GlobalPointer NER heads (logit thresholds)
    "ner_general": -0.8,  # F1=0.671 (P=0.786, R=0.586)
    "ner_family": -2.1,  # F1=0.730 (P=0.744, R=0.717)
    "temporal": -2.4,  # F1=0.640 (P=0.755, R=0.556)
    # LabelDescriptionHead (probability thresholds)
    "intent": 0.30,  # F1=0.821 @ temperature=0.10
    "ingress": 0.30,  # F1=0.774 @ temperature=0.10
}

# LabelDescriptionHead optimal temperatures (for reference - stored in model weights)
# These are the optimal temperatures from validation sweep:
LABEL_DESCRIPTION_TEMPERATURES = {
    "intent": 0.10,  # F1=0.821 (P=0.929, R=0.736)
    "ingress": 0.10,  # F1=0.774 (P=0.783, R=0.766)
}


# =============================================================================
# Result Dataclasses
# =============================================================================


from dataclasses import dataclass


@dataclass
class InferenceResult:
    """Result for a single capability."""

    capability: str
    output: Dict[str, Any]
    latency_ms: float


@dataclass
class AnalysisResult:
    """Result from multi-capability analysis."""

    text: str
    results: Dict[str, InferenceResult]
    total_latency_ms: float
    encoder_latency_ms: float
    heads_latency_ms: float
    from_cache: bool = False


# =============================================================================
# Cache
# =============================================================================


class EncoderCache:
    """LRU cache for encoder outputs."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict[str, tuple] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[tuple]:
        key = self._hash(text)
        with self._lock:
            if key in self.cache:
                self.hits += 1
                self.cache.move_to_end(key)
                return self.cache[key]
            self.misses += 1
            return None

    def put(self, text: str, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> None:
        key = self._hash(text)
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)
            self.cache[key] = (hidden_states.detach().clone(), attention_mask.detach().clone())

    def clear(self) -> None:
        with self._lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0.0,
        }


# =============================================================================
# Post-Processing
# =============================================================================


def _postprocess_token_classification(
    logits: torch.Tensor, tokens: List[str], schema: LabelSchema
) -> Dict[str, Any]:
    """Extract entities from token classification logits."""
    pred_ids = torch.argmax(logits, dim=-1)[0].cpu().numpy()
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


def _postprocess_sequence_single(logits: torch.Tensor, schema: LabelSchema) -> Dict[str, Any]:
    """Single-label sequence classification."""
    probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return {
        "prediction": schema.id2label[pred_idx],
        "confidence": round(float(probs[pred_idx]), 4),
        "scores": {schema.id2label[i]: round(float(p), 4) for i, p in enumerate(probs)},
    }


def _postprocess_sequence_multi(
    logits: torch.Tensor, schema: LabelSchema, threshold: float = 0.3
) -> Dict[str, Any]:
    """Multi-label sequence classification."""
    probs = torch.sigmoid(logits[0]).cpu().numpy()
    predictions = []
    scores = {}
    for i, p in enumerate(probs):
        label = schema.id2label[i]
        scores[label] = round(float(p), 4)
        if p >= threshold:
            predictions.append(label)
    return {"predictions": predictions, "scores": scores}


def _postprocess_embedding(logits: torch.Tensor) -> Dict[str, Any]:
    """Process embedding output."""
    embedding = logits[0].cpu().numpy()
    return {
        "embedding": embedding.tolist(),
        "dim": len(embedding),
        "norm": float(np.linalg.norm(embedding)),
    }


def _postprocess_safety(logits: torch.Tensor, schema: LabelSchema) -> Dict[str, Any]:
    """Safety band classification."""
    probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return {
        "band": schema.id2label[pred_idx],
        "confidence": round(float(probs[pred_idx]), 4),
        "probabilities": {schema.id2label[i]: round(float(p), 4) for i, p in enumerate(probs)},
    }


def _postprocess_label_description_intent(
    logits: torch.Tensor,
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
    probs = torch.sigmoid(logits[0]).cpu().numpy()
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
    logits: torch.Tensor,
    schema: LabelSchema,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Multi-label ingress (domain) classification with K1-compliant output.

    Returns:
        Dict with:
            - primary: Highest-scoring domain (always returned)
            - domains: List of domains above threshold
            - scores: Dict of domain -> probability
            - confidence: Probability of primary domain
    """
    probs = torch.sigmoid(logits[0]).cpu().numpy()
    scores = {schema.id2label[i]: round(float(p), 4) for i, p in enumerate(probs)}

    # Primary = highest scoring domain
    primary_idx = int(probs.argmax())
    primary = schema.id2label[primary_idx]
    confidence = round(float(probs[primary_idx]), 4)

    # Domains = all labels above threshold
    domains = [schema.id2label[i] for i, p in enumerate(probs) if p >= threshold]

    return {
        "primary": primary,
        "domains": domains,
        "scores": scores,
        "confidence": confidence,
    }


def _postprocess_globalpointer(
    logits: torch.Tensor,
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

    Uses vectorized operations for O(k) complexity where k = number of spans
    above threshold, instead of O(n^2) Python loops.

    Args:
        logits: (batch, num_labels, seq_len, seq_len) span scores
        text: Original input text for extracting entity strings
        offset_mapping: Token offset mapping from tokenizer
        schema: LabelSchema with id2label mapping
        threshold: Minimum logit threshold for entity detection (default: -1.0)

    Returns:
        Dict with 'entities' list containing detected spans
    """
    entities = []

    # Remove batch dimension: (num_labels, seq_len, seq_len)
    span_scores = logits[0]

    num_labels = span_scores.shape[0]
    seq_len = span_scores.shape[1]
    offset_len = len(offset_mapping)

    # Vectorized: find all positions above threshold using torch.where
    # Create upper triangular mask (start <= end)
    triu_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=span_scores.device)
    )

    for label_id in range(num_labels):
        label_scores = span_scores[label_id]

        # Apply mask and threshold in one operation
        masked_scores = label_scores.masked_fill(~triu_mask, float("-inf"))

        # Find positions above threshold (vectorized)
        above_thresh = masked_scores > threshold
        positions = torch.nonzero(above_thresh, as_tuple=False)

        if positions.shape[0] == 0:
            continue

        # Extract all spans at once
        tok_starts = positions[:, 0]
        tok_ends = positions[:, 1]
        logit_values = masked_scores[tok_starts, tok_ends]
        score_values = torch.sigmoid(logit_values)

        # Convert to Python and process
        for idx in range(positions.shape[0]):
            tok_start = tok_starts[idx].item()
            tok_end = tok_ends[idx].item()
            score = score_values[idx].item()

            # Bounds check
            if tok_start >= offset_len or tok_end >= offset_len:
                continue

            char_start, _ = offset_mapping[tok_start]
            _, char_end = offset_mapping[tok_end]

            # Skip special tokens (offsets are (0, 0))
            if char_start == 0 and char_end == 0 and tok_start > 0:
                continue

            entity_text = text[char_start:char_end]
            if not entity_text.strip():
                continue

            entities.append(
                {
                    "text": entity_text,
                    "label": schema.id2label[label_id],
                    "start": char_start,
                    "end": char_end,
                    "start_token": tok_start,
                    "end_token": tok_end,
                    "score": round(score, 4),
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
    logits: torch.Tensor,
    tokens: List[str],
    text: str = "",
    offset_mapping: Optional[List[tuple[int, int]]] = None,
    threshold: Optional[float] = None,
    custom_schema: Optional[LabelSchema] = None,
) -> Dict[str, Any]:
    """Post-process head output based on capability type.

    Args:
        capability: The capability name
        logits: Model output logits
        tokens: Tokenized input tokens (for BIO-based NER fallback)
        text: Original input text (for GlobalPointer span extraction)
        offset_mapping: Token offset mapping (for GlobalPointer)
        threshold: Override threshold for GlobalPointer (uses DEFAULT_THRESHOLDS if None)
        custom_schema: Optional custom schema (for zero-shot labels)

    Returns:
        Dict with processed predictions
    """
    cap_enum = Capability(capability)
    # Use custom schema if provided, otherwise fall back to global
    schema = custom_schema or CAPABILITY_TO_LABELS.get(cap_enum)

    if capability in ["ner_general", "ner_family", "temporal"]:
        # GlobalPointer: 4D output (batch, num_labels, seq, seq)
        if logits.dim() == 4 and offset_mapping is not None:
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
    elif capability == "relevance":
        # MGRH outputs are handled separately via score_relevance / rerank
        # If called through standard pipeline, return the raw score
        score = logits[0].item() if logits.dim() == 2 else logits.item()
        return {"score": round(float(score), 6)}
    elif capability == "safety_familyos":
        return _postprocess_safety(logits, schema)
    elif capability in ["emotions", "safety_generic", "relation"]:
        return _postprocess_sequence_multi(logits, schema)
    elif capability == "intent":
        # LabelDescriptionHead: multi-label intent
        thresh = threshold if threshold is not None else DEFAULT_THRESHOLDS.get(capability, 0.5)
        return _postprocess_label_description_intent(logits, schema, threshold=thresh)
    elif capability == "ingress":
        # LabelDescriptionHead: multi-label ingress
        thresh = threshold if threshold is not None else DEFAULT_THRESHOLDS.get(capability, 0.5)
        return _postprocess_label_description_ingress(logits, schema, threshold=thresh)
    else:
        return _postprocess_sequence_single(logits, schema)


# =============================================================================
# Model Loading
# =============================================================================


def _load_model(model_path: str, device: str):
    """Load the multi-task model using the bundled model classes."""
    from familyos_ultrabert.models.modernbert_multitask import ModernBertMultiTaskModel

    model = ModernBertMultiTaskModel.load_checkpoint(model_path, device=device)
    return model


# =============================================================================
# PyTorch Inference Engine
# =============================================================================


class PyTorchInferenceEngine:
    """
    High-performance PyTorch inference engine.

    Features:
    - Single encoder pass for multiple capabilities
    - Parallel head execution via CUDA streams
    - Per-head threshold configuration for GlobalPointer
    - Optional encoder output caching
    """

    def __init__(
        self,
        model: Any,
        tokenizer: AutoTokenizer,
        capabilities: List[str],
        device: str = "cuda",
        enable_cache: bool = True,
        cache_size: int = 1000,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.capabilities = capabilities
        self.device = device

        # Per-head thresholds (merge with defaults)
        self.thresholds = {**DEFAULT_THRESHOLDS}
        if thresholds:
            self.thresholds.update(thresholds)

        self.encoder = self.model.get_encoder()
        self.heads = self.model.heads

        self.cache = EncoderCache(cache_size) if enable_cache else None

        # Custom schemas for zero-shot labels (override CAPABILITY_TO_LABELS)
        self.custom_schemas: Dict[str, LabelSchema] = {}

        # MGRH calibration temperature (loaded from mgrh_metadata.json if present)
        self.mgrh_temperature: float = 1.0

        # CUDA streams for parallel head execution
        self.use_cuda = device.startswith("cuda") and torch.cuda.is_available()
        if self.use_cuda:
            self.streams = {cap: torch.cuda.Stream() for cap in capabilities}
        else:
            self.streams = {}

    @classmethod
    def load(
        cls,
        model_path: str,
        device: str = "auto",
        enable_cache: bool = True,
        cache_size: int = 1000,
    ) -> "PyTorchInferenceEngine":
        """Load engine from model directory."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = _load_model(model_path, device)
        tokenizer = _load_tokenizer(model_path)
        capabilities = list(model.heads.keys())

        engine = cls(
            model=model,
            tokenizer=tokenizer,
            capabilities=capabilities,
            device=device,
            enable_cache=enable_cache,
            cache_size=cache_size,
        )

        # Load MGRH calibration from metadata
        metadata_path = Path(model_path) / "mgrh_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                meta = json.load(f)
            cal = meta.get("calibration", {})
            engine.mgrh_temperature = cal.get("temperature", 1.0)
            logger.info("MGRH calibration temperature: %.4f", engine.mgrh_temperature)

            # MaxSim population z-norm stats (avoids batch-size-dependent scoring)
            pop_mean = cal.get("maxsim_population_mean")
            pop_std = cal.get("maxsim_population_std")
            if pop_mean is not None and pop_std is not None and "relevance" in engine.heads:
                engine.heads["relevance"]._maxsim_pop_mean = pop_mean
                engine.heads["relevance"]._maxsim_pop_std = pop_std
                logger.info(
                    "MGRH MaxSim population z-norm: mean=%.2f std=%.2f",
                    pop_mean, pop_std,
                )

        return engine

    def _encode(self, text: str, use_cache: bool = True) -> tuple:
        """Encode text to hidden states.

        Returns:
            Tuple of (hidden_states, attention_mask, tokens, offset_mapping, from_cache)
        """
        # Tokenize (always need offset_mapping for GlobalPointer)
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offset_mapping = inputs.pop("offset_mapping")[0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu().numpy())

        # Check cache
        if use_cache and self.cache:
            cached = self.cache.get(text)
            if cached:
                return cached[0], cached[1], tokens, offset_mapping, True

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Encode
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                return_dict=True,
            )
        hidden_states = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]

        # Cache
        if use_cache and self.cache:
            self.cache.put(text, hidden_states, attention_mask)

        return hidden_states, attention_mask, tokens, offset_mapping, False

    def _run_heads(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        capabilities: List[str],
        embedding_mode: str = DEFAULT_EMBEDDING_MODE,
    ) -> Dict[str, torch.Tensor]:
        """Run heads, extracting logits properly."""
        results = {}

        with torch.no_grad():
            for cap in capabilities:
                if cap not in self.heads:
                    continue
                # MGRH requires paired input; skip in standard single-text pipeline
                if cap == Capability.RELEVANCE.value:
                    continue
                head = self.heads[cap]

                if cap == Capability.EMBEDDING.value:
                    if embedding_mode not in {"query", "document"}:
                        raise ValueError(
                            "embedding_mode must be 'query' or 'document'"
                        )

                    if getattr(head, "pooling", None) == "agreement_gated_v2":
                        out = head(
                            hidden_states,
                            attention_mask=attention_mask,
                            mode=embedding_mode,
                        )
                    else:
                        out = head(hidden_states, attention_mask=attention_mask)
                else:
                    out = head(hidden_states, attention_mask=attention_mask)

                # Extract logits from dict output
                if isinstance(out, dict):
                    results[cap] = out.get("logits", out.get("embeddings"))
                else:
                    results[cap] = out

        return results

    @torch.no_grad()
    def analyze(
        self,
        text: str,
        capabilities: Optional[List[str]] = None,
        use_cache: bool = True,
        embedding_mode: str = DEFAULT_EMBEDDING_MODE,
    ) -> AnalysisResult:
        """Analyze text with multiple capabilities in a single pass.

        Args:
            text: Input text to analyze.
            capabilities: Capabilities to execute. None runs all available capabilities.
            use_cache: Whether to reuse cached encoder outputs.
            embedding_mode: Internal routing mode for retrieval embedding heads.
                Defaults to "document" to preserve current behavior.
        """
        start_time = time.perf_counter()

        # Resolve capabilities
        if capabilities is None:
            capabilities = self.capabilities
        else:
            capabilities = [c for c in capabilities if c in self.capabilities]

        if not capabilities:
            raise ValueError(f"No valid capabilities. Available: {self.capabilities}")

        # Encode (single pass)
        encode_start = time.perf_counter()
        hidden_states, attention_mask, tokens, offset_mapping, from_cache = self._encode(
            text, use_cache
        )
        encoder_ms = (time.perf_counter() - encode_start) * 1000

        # Run heads
        heads_start = time.perf_counter()
        head_outputs = self._run_heads(
            hidden_states,
            attention_mask,
            capabilities,
            embedding_mode=embedding_mode,
        )
        heads_ms = (time.perf_counter() - heads_start) * 1000

        # Post-process
        results = {}
        for cap in capabilities:
            if cap not in head_outputs:
                continue
            pp_start = time.perf_counter()
            # Use custom schema if set (for zero-shot labels)
            custom_schema = self.custom_schemas.get(cap)
            output = postprocess(
                cap,
                head_outputs[cap],
                tokens,
                text=text,
                offset_mapping=offset_mapping,
                threshold=self.thresholds.get(cap),
                custom_schema=custom_schema,
            )
            pp_ms = (time.perf_counter() - pp_start) * 1000
            results[cap] = InferenceResult(capability=cap, output=output, latency_ms=pp_ms)

        total_ms = (time.perf_counter() - start_time) * 1000

        return AnalysisResult(
            text=text,
            results=results,
            total_latency_ms=total_ms,
            encoder_latency_ms=encoder_ms,
            heads_latency_ms=heads_ms,
            from_cache=from_cache,
        )

    def clear_cache(self) -> None:
        """Clear encoder cache."""
        if self.cache:
            self.cache.clear()

    def cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.stats() if self.cache else {"enabled": False}

    # -----------------------------------------------------------------
    # MGRH Relevance Scoring
    # -----------------------------------------------------------------

    @torch.no_grad()
    def _encode_text(self, text: str) -> tuple:
        """Encode a single text and return (hidden_states, attention_mask)."""
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
        )
        return outputs.last_hidden_state, inputs["attention_mask"]

    @torch.no_grad()
    def _encode_pair_joint(self, query: str, doc: str) -> tuple:
        """Encode query+doc as a joint pair and return (hidden_states, attention_mask)."""
        inputs = self.tokenizer(
            query,
            doc,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
        )
        return outputs.last_hidden_state, inputs["attention_mask"]

    @torch.no_grad()
    def _get_embedding(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Get embedding from the embedding head (for MGRH Signal 3)."""
        if "embedding" not in self.heads:
            return None
        head = self.heads["embedding"]
        out = head(hidden_states, attention_mask=attention_mask)
        if isinstance(out, dict):
            return out.get("embedding", out.get("logits"))
        return out

    @torch.no_grad()
    def score_relevance(self, query: str, doc: str) -> Dict[str, Any]:
        """Score a (query, doc) pair using the MGRH head.

        Requires 3 encoder passes: joint, query-only, doc-only, plus embedding.

        Args:
            query: Query text.
            doc: Document text.

        Returns:
            Dict with 'score' (float 0-1), 'latency_ms', and raw signals.
        """
        if "relevance" not in self.heads:
            raise ValueError("MGRH (relevance) head not loaded. Check checkpoint.")

        start_time = time.perf_counter()

        # 3 encoder passes
        joint_hidden, joint_mask = self._encode_pair_joint(query, doc)
        q_hidden, q_mask = self._encode_text(query)
        d_hidden, d_mask = self._encode_text(doc)

        # Embedding signals
        q_embed = self._get_embedding(q_hidden, q_mask)
        d_embed = self._get_embedding(d_hidden, d_mask)

        # MGRH forward
        head = self.heads["relevance"]
        output = head(
            hidden_states=joint_hidden,
            attention_mask=joint_mask,
            text_a_hidden=q_hidden,
            text_b_hidden=d_hidden,
            text_a_mask=q_mask,
            text_b_mask=d_mask,
            query_embed=q_embed,
            doc_embed=d_embed,
            stage="c",
        )

        # Apply calibration temperature to raw logits
        raw_logit = output["relevance_logits_raw"].squeeze()
        score = torch.sigmoid(raw_logit / self.mgrh_temperature).item()
        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "score": round(score, 6),
            "latency_ms": round(latency_ms, 2),
        }

    @torch.no_grad()
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        batch_size: int = 16,
    ) -> List[Dict[str, Any]]:
        """Re-rank documents by MGRH relevance to query.

        Uses batched scoring so MaxSim z-normalization is correct.

        Args:
            query: Query text.
            documents: List of document texts.
            top_k: Return only top-k results. None returns all.
            batch_size: Documents per batch (>1 required for MaxSim z-norm).

        Returns:
            List of dicts sorted by relevance score (descending), each with
            'index' (original position), 'score', and 'text'.
        """
        if "relevance" not in self.heads:
            raise ValueError("MGRH (relevance) head not loaded. Check checkpoint.")

        head = self.heads["relevance"]
        all_scores: List[float] = []

        for start in range(0, len(documents), batch_size):
            chunk_docs = documents[start : start + batch_size]
            queries = [query] * len(chunk_docs)

            # Batched tokenization — pad to longest in batch (NOT max_length)
            # Using padding="max_length" floods short texts with 500+ pad tokens
            # whose hidden states leak through LayerNorm stats and corrupt scores.
            joint = self.tokenizer(
                queries, chunk_docs,
                max_length=512, padding=True,
                truncation=True, return_tensors="pt",
            )
            enc_q = self.tokenizer(
                queries,
                max_length=512, padding=True,
                truncation=True, return_tensors="pt",
            )
            enc_d = self.tokenizer(
                chunk_docs,
                max_length=512, padding=True,
                truncation=True, return_tensors="pt",
            )

            joint_ids = joint["input_ids"].to(self.device)
            joint_mask = joint["attention_mask"].to(self.device)
            q_ids = enc_q["input_ids"].to(self.device)
            q_mask = enc_q["attention_mask"].to(self.device)
            d_ids = enc_d["input_ids"].to(self.device)
            d_mask = enc_d["attention_mask"].to(self.device)

            # Encoder passes
            joint_hidden = self.encoder(
                input_ids=joint_ids, attention_mask=joint_mask, return_dict=True,
            ).last_hidden_state
            q_hidden = self.encoder(
                input_ids=q_ids, attention_mask=q_mask, return_dict=True,
            ).last_hidden_state
            d_hidden = self.encoder(
                input_ids=d_ids, attention_mask=d_mask, return_dict=True,
            ).last_hidden_state

            # Embeddings
            q_embed = self._get_embedding(q_hidden, q_mask)
            d_embed = self._get_embedding(d_hidden, d_mask)

            output = head(
                hidden_states=joint_hidden,
                attention_mask=joint_mask,
                text_a_hidden=q_hidden,
                text_b_hidden=d_hidden,
                text_a_mask=q_mask,
                text_b_mask=d_mask,
                query_embed=q_embed,
                doc_embed=d_embed,
                stage="c",
            )

            # Apply calibration temperature to raw logits
            raw_logits = output["relevance_logits_raw"].squeeze(-1).float()
            scores = torch.sigmoid(raw_logits / self.mgrh_temperature).cpu()
            if scores.dim() == 0:
                all_scores.append(scores.item())
            else:
                all_scores.extend(scores.tolist())

        results = [
            {"index": i, "score": round(s, 6), "text": documents[i]}
            for i, s in enumerate(all_scores)
        ]
        results.sort(key=lambda x: x["score"], reverse=True)
        if top_k is not None:
            results = results[:top_k]
        return results
