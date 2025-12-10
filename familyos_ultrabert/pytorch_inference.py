"""
FamilyOS UltraBERT v2 - PyTorch Inference Backend

High-performance inference using PyTorch with:
- Single encoder pass for multiple capabilities
- Parallel head execution via CUDA streams
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
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer

from .labels import CAPABILITY_TO_LABELS, Capability, LabelSchema

logger = logging.getLogger(__name__)


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


def postprocess(capability: str, logits: torch.Tensor, tokens: List[str]) -> Dict[str, Any]:
    """Post-process head output based on capability type."""
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
    ):
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.capabilities = capabilities
        self.device = device

        self.encoder = self.model.get_encoder()
        self.heads = self.model.heads

        self.cache = EncoderCache(cache_size) if enable_cache else None

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
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        capabilities = list(model.heads.keys())

        return cls(
            model=model,
            tokenizer=tokenizer,
            capabilities=capabilities,
            device=device,
            enable_cache=enable_cache,
            cache_size=cache_size,
        )

    def _encode(self, text: str, use_cache: bool = True) -> tuple:
        """Encode text to hidden states."""
        # Check cache
        if use_cache and self.cache:
            cached = self.cache.get(text)
            if cached:
                tokens = self.tokenizer.tokenize(text)
                return cached[0], cached[1], tokens, True

        # Tokenize
        inputs = self.tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu().numpy())

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

        return hidden_states, attention_mask, tokens, False

    def _run_heads(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, capabilities: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Run heads, extracting logits properly."""
        results = {}

        with torch.no_grad():
            for cap in capabilities:
                if cap not in self.heads:
                    continue
                head = self.heads[cap]
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
    ) -> AnalysisResult:
        """Analyze text with multiple capabilities in a single pass."""
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
        hidden_states, attention_mask, tokens, from_cache = self._encode(text, use_cache)
        encoder_ms = (time.perf_counter() - encode_start) * 1000

        # Run heads
        heads_start = time.perf_counter()
        head_outputs = self._run_heads(hidden_states, attention_mask, capabilities)
        heads_ms = (time.perf_counter() - heads_start) * 1000

        # Post-process
        results = {}
        for cap in capabilities:
            if cap not in head_outputs:
                continue
            pp_start = time.perf_counter()
            output = postprocess(cap, head_outputs[cap], tokens)
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
