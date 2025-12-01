#!/usr/bin/env python
"""
Optimized Multi-Task Inference

This module provides a highly optimized inference engine for the multi-task model
that minimizes latency through:
- Single forward pass through encoder (shared across all capabilities)
- Parallel head execution via torch.compile or threading
- Encoder output caching for similar/repeated inputs
- Lazy head execution (only compute requested capabilities)
- Batched head computation for maximum throughput

Key Optimization Strategies:
1. **Single Encoder Pass**: Run encoder once, reuse hidden states for all heads
2. **Parallel Heads**: Execute multiple heads concurrently (GPU parallel or threaded)
3. **Caching**: LRU cache for encoder outputs (useful for repeated/similar inputs)
4. **Lazy Execution**: Only run heads that are actually requested
5. **Compiled Inference**: Use torch.compile for optimized execution

Performance Targets:
- 12 capabilities in < 15ms (vs 12 × 10ms = 120ms sequential)
- Throughput: > 1000 samples/second on A100 GPU

Usage:
    from export_utility.optimized_inference import OptimizedMultiTaskModel

    # Load optimized model
    model = OptimizedMultiTaskModel.from_pretrained(
        "outputs/modernbert-multitask-v0",
        enable_caching=True,
        parallel_heads=True,
    )

    # Single inference with multiple capabilities
    outputs = model.infer(
        text="Mom picked up Panda from school",
        capabilities=["ner_family", "sentiment", "safety_familyos", "intent"],
    )

    # Batch inference
    outputs = model.infer_batch(
        texts=["text1", "text2", "text3"],
        capabilities=["sentiment", "safety_familyos"],
    )

    # With caching (useful for API servers)
    outputs = model.infer(
        text="Same text again",
        capabilities=["sentiment"],
        use_cache=True,
    )
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modeling_studio.data.labels import CAPABILITY_TO_LABELS, Capability
from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Output Structures
# =============================================================================


@dataclass
class InferenceResult:
    """Result from a single capability inference."""

    capability: str
    output: dict[str, Any]
    latency_ms: float
    from_cache: bool = False


@dataclass
class MultiCapabilityResult:
    """Result from multi-capability inference."""

    text: str
    results: dict[str, InferenceResult]
    encoder_latency_ms: float
    heads_latency_ms: float
    total_latency_ms: float
    num_capabilities: int
    from_cache: bool = False


@dataclass
class BatchInferenceResult:
    """Result from batch inference."""

    results: list[MultiCapabilityResult]
    total_samples: int
    total_latency_ms: float
    throughput_samples_per_sec: float
    avg_latency_per_sample_ms: float


# =============================================================================
# Encoder Output Cache
# =============================================================================


class EncoderCache:
    """
    LRU cache for encoder outputs.

    Uses text hash as key to cache hidden states, avoiding
    redundant encoder forward passes for repeated inputs.
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of cached entries
        """
        self.max_size = max_size
        self.cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _hash_text(self, text: str) -> str:
        """Create a hash key for text."""
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        """
        Get cached encoder output.

        Args:
            text: Input text

        Returns:
            Tuple of (hidden_states, attention_mask) or None if not cached
        """
        key = self._hash_text(text)
        with self._lock:
            if key in self.cache:
                self.hits += 1
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
            self.misses += 1
            return None

    def put(
        self,
        text: str,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> None:
        """
        Store encoder output in cache.

        Args:
            text: Input text
            hidden_states: Encoder hidden states
            attention_mask: Attention mask
        """
        key = self._hash_text(text)
        with self._lock:
            if key in self.cache:
                # Update existing
                self.cache.move_to_end(key)
            else:
                # Add new
                if len(self.cache) >= self.max_size:
                    # Remove oldest
                    self.cache.popitem(last=False)
            # Store copies to avoid tensor reference issues
            self.cache[key] = (
                hidden_states.detach().clone(),
                attention_mask.detach().clone(),
            )

    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0.0,
        }


# =============================================================================
# Parallel Head Executor
# =============================================================================


class ParallelHeadExecutor:
    """
    Execute multiple heads in parallel.

    Uses GPU parallelism via CUDA streams or CPU threading
    to run heads concurrently.
    """

    def __init__(
        self,
        heads: nn.ModuleDict,
        device: str = "cuda",
        use_cuda_streams: bool = True,
        max_workers: int = 4,
    ):
        """
        Initialize parallel executor.

        Args:
            heads: ModuleDict of task heads
            device: Device for execution
            use_cuda_streams: Use CUDA streams for GPU parallelism
            max_workers: Max threads for CPU parallelism
        """
        self.heads = heads
        self.device = device
        self.use_cuda_streams = use_cuda_streams and device.startswith("cuda")
        self.max_workers = max_workers

        # Create CUDA streams for parallel execution
        if self.use_cuda_streams:
            self.streams = {cap: torch.cuda.Stream() for cap in heads.keys()}
        else:
            self.streams = {}

        # Thread pool for CPU parallel execution
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)

    def execute_parallel(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        capabilities: list[str],
    ) -> dict[str, torch.Tensor]:
        """
        Execute multiple heads in parallel.

        Args:
            hidden_states: Encoder output (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask (batch_size, seq_len)
            capabilities: List of capability names to execute

        Returns:
            Dict mapping capability name to output logits
        """
        if self.use_cuda_streams:
            return self._execute_cuda_streams(hidden_states, attention_mask, capabilities)
        else:
            return self._execute_threaded(hidden_states, attention_mask, capabilities)

    def _execute_cuda_streams(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        capabilities: list[str],
    ) -> dict[str, torch.Tensor]:
        """Execute heads in parallel using CUDA streams."""
        results = {}

        # Launch all heads in parallel streams
        for cap in capabilities:
            if cap not in self.heads:
                continue
            stream = self.streams.get(cap)
            if stream is None:
                stream = torch.cuda.Stream()
                self.streams[cap] = stream

            with torch.cuda.stream(stream):
                head = self.heads[cap]
                # Execute head
                if hasattr(head, "forward_with_mask"):
                    results[cap] = head.forward_with_mask(hidden_states, attention_mask)
                else:
                    results[cap] = head(hidden_states, attention_mask=attention_mask)

        # Synchronize all streams
        torch.cuda.synchronize()

        return results

    def _execute_threaded(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        capabilities: list[str],
    ) -> dict[str, torch.Tensor]:
        """Execute heads in parallel using thread pool (for CPU)."""

        def run_head(cap: str) -> tuple[str, torch.Tensor]:
            if cap not in self.heads:
                return cap, None
            head = self.heads[cap]
            with torch.no_grad():
                if hasattr(head, "forward_with_mask"):
                    output = head.forward_with_mask(hidden_states, attention_mask)
                else:
                    output = head(hidden_states, attention_mask=attention_mask)
            return cap, output

        results = {}
        futures = [self.thread_pool.submit(run_head, cap) for cap in capabilities]

        for future in futures:
            cap, output = future.result()
            if output is not None:
                results[cap] = output

        return results

    def shutdown(self) -> None:
        """Shutdown thread pool."""
        self.thread_pool.shutdown(wait=False)


# =============================================================================
# Post-Processing Functions
# =============================================================================


def postprocess_token_classification(
    logits: torch.Tensor,
    tokens: list[str],
    labels_schema: Any,
) -> dict:
    """Post-process token classification (NER, temporal)."""
    pred_ids = torch.argmax(logits, dim=-1)[0].cpu().numpy()
    pred_labels = [labels_schema.id2label[int(i)] for i in pred_ids]

    # Extract entities
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
            entity_type = label[2:]
            current_entity = {
                "text": token.replace("##", "").replace("Ġ", " ").strip(),
                "label": entity_type,
                "start_token": i,
                "end_token": i,
            }
        elif label.startswith("I-") and current_entity:
            entity_type = label[2:]
            if entity_type == current_entity["label"]:
                current_entity["text"] += token.replace("##", "").replace("Ġ", " ")
                current_entity["end_token"] = i
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    return {"entities": entities}


def postprocess_sequence_classification(
    logits: torch.Tensor,
    labels_schema: Any,
    multi_label: bool = False,
    threshold: float = 0.3,
) -> dict:
    """Post-process sequence classification."""
    if multi_label:
        probs = torch.sigmoid(logits[0]).cpu().numpy()
        predictions = []
        scores = {}
        for i, p in enumerate(probs):
            label = labels_schema.id2label[i]
            scores[label] = round(float(p), 4)
            if p >= threshold:
                predictions.append(label)
        return {"predictions": predictions, "scores": scores}
    else:
        probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
        pred_idx = int(np.argmax(probs))
        return {
            "prediction": labels_schema.id2label[pred_idx],
            "confidence": round(float(probs[pred_idx]), 4),
            "scores": {labels_schema.id2label[i]: round(float(p), 4) for i, p in enumerate(probs)},
        }


def postprocess_embedding(logits: torch.Tensor) -> dict:
    """Post-process embedding output."""
    embedding = logits[0].cpu().numpy()
    return {
        "embedding": embedding.tolist(),
        "dim": len(embedding),
        "norm": float(np.linalg.norm(embedding)),
    }


def postprocess_safety(logits: torch.Tensor, labels_schema: Any) -> dict:
    """Post-process safety band prediction."""
    probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return {
        "band": labels_schema.id2label[pred_idx],
        "confidence": round(float(probs[pred_idx]), 4),
        "probabilities": {
            labels_schema.id2label[i]: round(float(p), 4) for i, p in enumerate(probs)
        },
    }


# =============================================================================
# Optimized Multi-Task Model
# =============================================================================


class OptimizedMultiTaskModel:
    """
    Highly optimized multi-task inference model.

    Features:
    - Single encoder pass for multiple capabilities
    - Parallel head execution
    - Encoder output caching
    - Lazy head execution
    - Optional torch.compile optimization

    Example:
        >>> model = OptimizedMultiTaskModel.from_pretrained(
        ...     "outputs/modernbert-multitask-v0",
        ...     enable_caching=True,
        ...     parallel_heads=True,
        ... )
        >>> results = model.infer(
        ...     "Sample text",
        ...     capabilities=["sentiment", "safety_familyos", "intent"],
        ... )
    """

    def __init__(
        self,
        model: ModernBertMultiTaskModel,
        tokenizer: AutoTokenizer,
        device: str = "auto",
        enable_caching: bool = True,
        cache_size: int = 1000,
        parallel_heads: bool = True,
        use_compile: bool = False,
        compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default",
    ):
        """
        Initialize optimized model.

        Args:
            model: Loaded multi-task model
            tokenizer: Tokenizer for model
            device: Device (auto, cpu, cuda, cuda:0)
            enable_caching: Enable encoder output caching
            cache_size: Maximum cache entries
            parallel_heads: Enable parallel head execution
            use_compile: Use torch.compile for optimization (requires PyTorch 2.0+)
            compile_mode: torch.compile mode
        """
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Store model and tokenizer
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer

        # Get encoder reference
        self.encoder = self.model.get_encoder()
        self.heads = self.model.heads

        # Available capabilities
        self.capabilities = list(self.heads.keys())

        # Caching
        self.enable_caching = enable_caching
        self.cache = EncoderCache(max_size=cache_size) if enable_caching else None

        # Parallel execution
        self.parallel_heads = parallel_heads
        if parallel_heads:
            self.head_executor = ParallelHeadExecutor(
                heads=self.heads,
                device=self.device,
                use_cuda_streams=self.device.startswith("cuda"),
            )
        else:
            self.head_executor = None

        # Optional torch.compile
        self.use_compile = use_compile
        if use_compile and hasattr(torch, "compile"):
            logger.info(f"Compiling encoder with mode: {compile_mode}")
            self.encoder = torch.compile(self.encoder, mode=compile_mode)
            # Compile heads
            for cap_name in self.heads:
                self.heads[cap_name] = torch.compile(self.heads[cap_name], mode=compile_mode)

        # Task adapters (if available)
        self.task_adapters = getattr(self.model, "task_adapters", None)

        logger.info("OptimizedMultiTaskModel initialized:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Capabilities: {self.capabilities}")
        logger.info(f"  Caching: {enable_caching} (max {cache_size})")
        logger.info(f"  Parallel heads: {parallel_heads}")
        logger.info(f"  Compiled: {use_compile}")

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "auto",
        enable_caching: bool = True,
        cache_size: int = 1000,
        parallel_heads: bool = True,
        use_compile: bool = False,
        compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default",
    ) -> OptimizedMultiTaskModel:
        """
        Load optimized model from checkpoint.

        Args:
            model_path: Path to model checkpoint
            device: Device to use
            enable_caching: Enable encoder output caching
            cache_size: Max cache entries
            parallel_heads: Enable parallel head execution
            use_compile: Use torch.compile
            compile_mode: Compile mode

        Returns:
            OptimizedMultiTaskModel instance
        """
        logger.info(f"Loading model from {model_path}...")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load model
        actual_device = device
        if actual_device == "auto":
            actual_device = "cuda" if torch.cuda.is_available() else "cpu"

        model = ModernBertMultiTaskModel.load_checkpoint(model_path, device=actual_device)

        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            enable_caching=enable_caching,
            cache_size=cache_size,
            parallel_heads=parallel_heads,
            use_compile=use_compile,
            compile_mode=compile_mode,
        )

    def _encode(
        self,
        text: str,
        use_cache: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str], bool]:
        """
        Encode text to hidden states.

        Args:
            text: Input text
            use_cache: Whether to use cache

        Returns:
            Tuple of (hidden_states, attention_mask, tokens, from_cache)
        """
        from_cache = False

        # Check cache
        if use_cache and self.cache is not None:
            cached = self.cache.get(text)
            if cached is not None:
                hidden_states, attention_mask = cached
                tokens = self.tokenizer.tokenize(text)
                return hidden_states, attention_mask, tokens, True

        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get tokens for post-processing
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu().numpy())

        # Encode
        with torch.no_grad():
            encoder_outputs = self.encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                return_dict=True,
            )

        hidden_states = encoder_outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]

        # Store in cache
        if use_cache and self.cache is not None:
            self.cache.put(text, hidden_states, attention_mask)

        return hidden_states, attention_mask, tokens, from_cache

    def _run_heads(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        capabilities: list[str],
    ) -> dict[str, torch.Tensor]:
        """
        Run task heads on encoded input.

        Args:
            hidden_states: Encoder output
            attention_mask: Attention mask
            capabilities: Capabilities to execute

        Returns:
            Dict mapping capability to logits
        """
        # Apply task adapters if available
        if self.task_adapters is not None:
            # Group capabilities by task group and apply adapters
            # For simplicity, we'll apply adapter per capability
            pass  # Adapter is applied per-head in the forward pass

        if self.parallel_heads and self.head_executor is not None:
            return self.head_executor.execute_parallel(hidden_states, attention_mask, capabilities)
        else:
            # Sequential execution
            results = {}
            with torch.no_grad():
                for cap in capabilities:
                    if cap not in self.heads:
                        continue
                    head = self.heads[cap]
                    results[cap] = head(hidden_states, attention_mask=attention_mask)
            return results

    def _postprocess(
        self,
        capability: str,
        logits: torch.Tensor,
        tokens: list[str],
    ) -> dict:
        """Post-process head output based on capability type."""
        labels_schema = CAPABILITY_TO_LABELS.get(Capability(capability))

        # Token classification
        if capability in ["ner_general", "ner_family", "temporal"]:
            return postprocess_token_classification(logits, tokens, labels_schema)

        # Embedding
        elif capability == "embedding":
            return postprocess_embedding(logits)

        # Safety FamilyOS
        elif capability == "safety_familyos":
            return postprocess_safety(logits, labels_schema)

        # Multi-label (emotions, safety_generic)
        elif capability in ["emotions", "safety_generic"]:
            return postprocess_sequence_classification(logits, labels_schema, multi_label=True)

        # Single-label classification
        else:
            return postprocess_sequence_classification(logits, labels_schema, multi_label=False)

    @torch.no_grad()
    def infer(
        self,
        text: str,
        capabilities: list[str] | None = None,
        use_cache: bool = True,
    ) -> MultiCapabilityResult:
        """
        Run inference for multiple capabilities in a single pass.

        This is the main optimization - a single encoder pass is shared
        across all requested capabilities.

        Args:
            text: Input text
            capabilities: List of capabilities (None = all)
            use_cache: Use encoder cache

        Returns:
            MultiCapabilityResult with all outputs
        """
        start_time = time.perf_counter()

        # Default to all capabilities
        if capabilities is None:
            capabilities = self.capabilities
        else:
            # Filter to available capabilities
            capabilities = [c for c in capabilities if c in self.capabilities]

        if not capabilities:
            raise ValueError(f"No valid capabilities. Available: {self.capabilities}")

        # Single encoder pass (or cache hit)
        encode_start = time.perf_counter()
        hidden_states, attention_mask, tokens, from_cache = self._encode(text, use_cache=use_cache)
        encoder_latency_ms = (time.perf_counter() - encode_start) * 1000

        # Run all heads (parallel or sequential)
        heads_start = time.perf_counter()
        head_outputs = self._run_heads(hidden_states, attention_mask, capabilities)
        heads_latency_ms = (time.perf_counter() - heads_start) * 1000

        # Post-process results
        results = {}
        for cap in capabilities:
            if cap not in head_outputs:
                continue
            logits = head_outputs[cap]

            postprocess_start = time.perf_counter()
            output = self._postprocess(cap, logits, tokens)
            postprocess_latency = (time.perf_counter() - postprocess_start) * 1000

            results[cap] = InferenceResult(
                capability=cap,
                output=output,
                latency_ms=postprocess_latency,
                from_cache=from_cache,
            )

        total_latency_ms = (time.perf_counter() - start_time) * 1000

        return MultiCapabilityResult(
            text=text,
            results=results,
            encoder_latency_ms=encoder_latency_ms,
            heads_latency_ms=heads_latency_ms,
            total_latency_ms=total_latency_ms,
            num_capabilities=len(capabilities),
            from_cache=from_cache,
        )

    @torch.no_grad()
    def infer_batch(
        self,
        texts: list[str],
        capabilities: list[str] | None = None,
        use_cache: bool = True,
        batch_size: int = 32,
    ) -> BatchInferenceResult:
        """
        Run batch inference for multiple texts and capabilities.

        Args:
            texts: List of input texts
            capabilities: Capabilities to run
            use_cache: Use encoder cache
            batch_size: Batch size for processing

        Returns:
            BatchInferenceResult with all outputs
        """
        start_time = time.perf_counter()
        results = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Process each text in batch
            for text in batch_texts:
                result = self.infer(text, capabilities, use_cache=use_cache)
                results.append(result)

        total_latency_ms = (time.perf_counter() - start_time) * 1000
        throughput = len(texts) / (total_latency_ms / 1000) if total_latency_ms > 0 else 0
        avg_latency = total_latency_ms / len(texts) if texts else 0

        return BatchInferenceResult(
            results=results,
            total_samples=len(texts),
            total_latency_ms=total_latency_ms,
            throughput_samples_per_sec=throughput,
            avg_latency_per_sample_ms=avg_latency,
        )

    def clear_cache(self) -> None:
        """Clear the encoder cache."""
        if self.cache is not None:
            self.cache.clear()

    def cache_stats(self) -> dict:
        """Get cache statistics."""
        if self.cache is not None:
            return self.cache.stats()
        return {"enabled": False}

    def benchmark(
        self,
        text: str = "This is a sample text for benchmarking the model performance.",
        capabilities: list[str] | None = None,
        num_warmup: int = 5,
        num_runs: int = 100,
    ) -> dict:
        """
        Benchmark inference performance.

        Args:
            text: Sample text
            capabilities: Capabilities to benchmark
            num_warmup: Warmup runs
            num_runs: Measurement runs

        Returns:
            Benchmark results dict
        """
        if capabilities is None:
            capabilities = self.capabilities

        # Warmup
        logger.info(f"Warming up ({num_warmup} runs)...")
        for _ in range(num_warmup):
            self.infer(text, capabilities, use_cache=False)

        # Clear cache for fair measurement
        self.clear_cache()

        # Benchmark without cache
        logger.info(f"Benchmarking without cache ({num_runs} runs)...")
        latencies_no_cache = []
        for _ in range(num_runs):
            result = self.infer(text, capabilities, use_cache=False)
            latencies_no_cache.append(result.total_latency_ms)

        # Benchmark with cache (second run should hit cache)
        logger.info(f"Benchmarking with cache ({num_runs} runs)...")
        self.clear_cache()
        latencies_with_cache = []
        for i in range(num_runs):
            result = self.infer(text, capabilities, use_cache=True)
            if i > 0:  # First run populates cache
                latencies_with_cache.append(result.total_latency_ms)

        def percentile(data: list, p: float) -> float:
            sorted_data = sorted(data)
            idx = int(len(sorted_data) * p / 100)
            return sorted_data[min(idx, len(sorted_data) - 1)]

        return {
            "text_length": len(text),
            "num_capabilities": len(capabilities),
            "capabilities": capabilities,
            "num_runs": num_runs,
            "without_cache": {
                "mean_ms": np.mean(latencies_no_cache),
                "std_ms": np.std(latencies_no_cache),
                "p50_ms": percentile(latencies_no_cache, 50),
                "p90_ms": percentile(latencies_no_cache, 90),
                "p99_ms": percentile(latencies_no_cache, 99),
                "min_ms": min(latencies_no_cache),
                "max_ms": max(latencies_no_cache),
            },
            "with_cache": {
                "mean_ms": np.mean(latencies_with_cache) if latencies_with_cache else 0,
                "std_ms": np.std(latencies_with_cache) if latencies_with_cache else 0,
                "p50_ms": percentile(latencies_with_cache, 50) if latencies_with_cache else 0,
                "p90_ms": percentile(latencies_with_cache, 90) if latencies_with_cache else 0,
                "p99_ms": percentile(latencies_with_cache, 99) if latencies_with_cache else 0,
            },
            "speedup_from_cache": (
                np.mean(latencies_no_cache) / np.mean(latencies_with_cache)
                if latencies_with_cache and np.mean(latencies_with_cache) > 0
                else 0
            ),
        }


# =============================================================================
# CLI
# =============================================================================


def main():
    """Command-line interface for optimized inference."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimized Multi-Task Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single text, multiple capabilities
  python export_utility/optimized_inference.py \\
      --model outputs/modernbert-multitask-v0 \\
      --text "Mom picked up Panda from school" \\
      --capabilities ner_family sentiment safety_familyos intent

  # Benchmark mode
  python export_utility/optimized_inference.py \\
      --model outputs/modernbert-multitask-v0 \\
      --benchmark \\
      --capabilities all

  # Interactive mode
  python export_utility/optimized_inference.py \\
      --model outputs/modernbert-multitask-v0 \\
      --interactive
""",
    )

    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Path to model directory",
    )
    parser.add_argument(
        "--text",
        "-t",
        help="Text to analyze",
    )
    parser.add_argument(
        "--capabilities",
        "-c",
        nargs="+",
        default=["all"],
        help="Capabilities to run (or 'all')",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark mode",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=100,
        help="Number of benchmark runs",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable encoder caching",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel head execution",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile (PyTorch 2.0+)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device (auto, cpu, cuda)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file for results (JSON)",
    )

    args = parser.parse_args()

    # Load model
    logger.info("Loading optimized model...")
    model = OptimizedMultiTaskModel.from_pretrained(
        args.model,
        device=args.device,
        enable_caching=not args.no_cache,
        parallel_heads=not args.no_parallel,
        use_compile=args.compile,
    )

    # Resolve capabilities
    if "all" in args.capabilities:
        capabilities = model.capabilities
    else:
        capabilities = args.capabilities

    # Benchmark mode
    if args.benchmark:
        text = args.text or "This is a sample text for benchmarking the model performance."
        results = model.benchmark(
            text=text,
            capabilities=capabilities,
            num_runs=args.benchmark_runs,
        )

        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Text length: {results['text_length']} chars")
        print(f"Capabilities: {results['num_capabilities']}")
        print(f"Runs: {results['num_runs']}")
        print()
        print("Without Cache:")
        print(f"  Mean: {results['without_cache']['mean_ms']:.2f} ms")
        print(f"  P50:  {results['without_cache']['p50_ms']:.2f} ms")
        print(f"  P90:  {results['without_cache']['p90_ms']:.2f} ms")
        print(f"  P99:  {results['without_cache']['p99_ms']:.2f} ms")
        print()
        print("With Cache:")
        print(f"  Mean: {results['with_cache']['mean_ms']:.2f} ms")
        print(f"  P50:  {results['with_cache']['p50_ms']:.2f} ms")
        print(f"  P99:  {results['with_cache']['p99_ms']:.2f} ms")
        print(f"\nCache Speedup: {results['speedup_from_cache']:.1f}x")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")

        return

    # Interactive mode
    if args.interactive:
        print("\n" + "=" * 60)
        print("OPTIMIZED MULTI-TASK INFERENCE")
        print("=" * 60)
        print(f"Capabilities: {capabilities}")
        print("Enter text (or 'quit' to exit):")
        print()

        while True:
            try:
                text = input(">>> ").strip()
                if text.lower() in ("quit", "exit", "q"):
                    break
                if not text:
                    continue

                result = model.infer(text, capabilities)

                print(
                    f"\n[{result.total_latency_ms:.1f}ms total, "
                    f"encoder: {result.encoder_latency_ms:.1f}ms, "
                    f"heads: {result.heads_latency_ms:.1f}ms, "
                    f"cache: {'HIT' if result.from_cache else 'MISS'}]"
                )
                print()

                for cap, res in result.results.items():
                    print(f"  {cap}:")
                    for key, value in res.output.items():
                        if key == "embedding":
                            print(f"    {key}: <{len(value)}-dim vector>")
                        elif key == "scores" and len(value) > 5:
                            print(f"    {key}: <{len(value)} labels>")
                        else:
                            print(f"    {key}: {value}")
                print()

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

        # Print cache stats
        print("\nCache stats:", model.cache_stats())
        return

    # Single inference
    if args.text:
        result = model.infer(args.text, capabilities)

        print("\n" + "=" * 60)
        print("INFERENCE RESULTS")
        print("=" * 60)
        print(f"Text: {args.text[:100]}{'...' if len(args.text) > 100 else ''}")
        print(f"Total latency: {result.total_latency_ms:.2f} ms")
        print(f"  Encoder: {result.encoder_latency_ms:.2f} ms")
        print(f"  Heads: {result.heads_latency_ms:.2f} ms")
        print(f"Cache: {'HIT' if result.from_cache else 'MISS'}")
        print()

        output_data = {"text": args.text, "results": {}}

        for cap, res in result.results.items():
            print(f"\n{cap}:")
            output_data["results"][cap] = res.output
            for key, value in res.output.items():
                if key == "embedding":
                    print(f"  {key}: <{len(value)}-dim vector>")
                else:
                    print(
                        f"  {key}: {json.dumps(value, indent=4) if isinstance(value, (dict, list)) else value}"
                    )

        if args.output:
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to: {args.output}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
    main()
