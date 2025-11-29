"""
Benchmark Suite

This module provides standardized benchmarks for comparing models
and tracking progress over time.

Classes:
    LatencyBenchmark: Measure inference latency with warmup and percentile reporting
    BenchmarkSuite: Full benchmark suite for model comparison (TODO)

Benchmarks:
    Generic NLU:
        - GLUE subset (SST-2, MNLI, QQP, etc.)
        - CoNLL-2003 NER
        - GoEmotions

    Safety:
        - Jigsaw toxicity
        - Civil Comments

    Embedding:
        - STS Benchmark
        - Retrieval benchmarks

    FamilyOS-specific:
        - Family NER test set
        - Ingress classification test set
        - Safety policy bands test set

Comparison Baselines:
    - BERT-base
    - DeBERTa-v3-base
    - Current zoo models (before unification)
    - ModernBERT-base (vanilla)

Usage:
    # Latency benchmarking
    from modeling_studio.evaluation.benchmarks import LatencyBenchmark

    benchmark = LatencyBenchmark(model=model, tokenizer=tokenizer)
    results = benchmark.run(
        texts=["Sample text"] * 100,
        batch_size=1,
        warmup=10,
        capability="sentiment",
    )
    print(f"P50={results['p50_ms']:.1f}ms, P95={results['p95_ms']:.1f}ms")

    # Full benchmark suite (TODO)
    benchmark = BenchmarkSuite(
        model=model,
        baselines=["bert-base", "deberta-v3-base"],
    )
    results = benchmark.run_all()
    benchmark.generate_comparison_table()
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


__all__ = [
    "LatencyBenchmark",
    "LatencyResults",
    "BenchmarkSuite",
    "BenchmarkResults",
    "BaseBenchmark",
    "GLUEBenchmark",
    "NERBenchmark",
    "EmbeddingBenchmark",
    "FamilyOSBenchmark",
    "BaselineComparison",
    "BenchmarkResultTracker",
]


@dataclass
class LatencyResults:
    """Results from latency benchmarking.

    Attributes:
        p50_ms: 50th percentile (median) latency in milliseconds
        p95_ms: 95th percentile latency in milliseconds
        p99_ms: 99th percentile latency in milliseconds
        mean_ms: Mean latency in milliseconds
        std_ms: Standard deviation of latency in milliseconds
        min_ms: Minimum latency in milliseconds
        max_ms: Maximum latency in milliseconds
        memory_mb: Peak GPU memory usage in megabytes (0 if CPU)
        throughput: Samples per second
        num_samples: Number of samples measured
        batch_size: Batch size used
        capability: Capability tested
        device: Device used for inference
        latencies_ms: Raw latency measurements in milliseconds
    """

    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    memory_mb: float
    throughput: float
    num_samples: int
    batch_size: int
    capability: str
    device: str
    latencies_ms: list[float] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert results to dictionary."""
        return {
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "memory_mb": self.memory_mb,
            "throughput": self.throughput,
            "num_samples": self.num_samples,
            "batch_size": self.batch_size,
            "capability": self.capability,
            "device": self.device,
        }

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            f"Latency Benchmark Results ({self.capability})",
            "=" * 50,
            f"Device: {self.device}",
            f"Samples: {self.num_samples}, Batch Size: {self.batch_size}",
            "",
            "Latency (ms):",
            f"  P50 (median): {self.p50_ms:.2f}",
            f"  P95:          {self.p95_ms:.2f}",
            f"  P99:          {self.p99_ms:.2f}",
            f"  Mean ± Std:   {self.mean_ms:.2f} ± {self.std_ms:.2f}",
            f"  Min / Max:    {self.min_ms:.2f} / {self.max_ms:.2f}",
            "",
            f"Throughput: {self.throughput:.1f} samples/sec",
            f"Memory: {self.memory_mb:.1f} MB",
        ]
        return "\n".join(lines)


class LatencyBenchmark:
    """Benchmark inference latency with warmup and percentile reporting.

    This class measures per-sample inference latency with configurable warmup
    runs, batch sizes, and capabilities. It reports percentile metrics (P50,
    P95, P99) and memory usage.

    Args:
        model: The model to benchmark (ModernBertMultiTaskModel or similar)
        tokenizer: Tokenizer for text preprocessing
        device: Device to run inference on ('cuda', 'cpu', or None for auto)

    Example:
        >>> benchmark = LatencyBenchmark(model=model, tokenizer=tokenizer)
        >>> results = benchmark.run(
        ...     texts=["Sample text " * 10] * 100,
        ...     batch_size=1,
        ...     warmup=10,
        ...     capability="sentiment",
        ... )
        >>> print(f"P50={results['p50_ms']:.1f}ms, P95={results['p95_ms']:.1f}ms")
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str | None = None,
    ) -> None:
        """Initialize latency benchmark.

        Args:
            model: Model to benchmark
            tokenizer: Tokenizer for preprocessing
            device: Device for inference (auto-detected if None)
        """
        self.model = model
        self.tokenizer = tokenizer

        # Determine device
        if device is None:
            if hasattr(model, "device"):
                self.device = str(model.device)
            elif next(model.parameters(), None) is not None:
                self.device = str(next(model.parameters()).device)
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Move model to device if needed
        if self.device != "cpu":
            self.model = self.model.to(self.device)  # type: ignore[arg-type]

    def _get_memory_mb(self) -> float:
        """Get current GPU memory usage in MB."""
        if not torch.cuda.is_available() or "cpu" in self.device:
            return 0.0

        try:
            torch.cuda.synchronize()
            memory_bytes = torch.cuda.max_memory_allocated(self.device)
            return memory_bytes / (1024 * 1024)
        except Exception:
            return 0.0

    def _reset_memory_stats(self) -> None:
        """Reset GPU memory statistics."""
        if torch.cuda.is_available() and "cpu" not in self.device:
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
            except Exception:
                pass

    def _tokenize_batch(
        self,
        texts: list[str],
        max_length: int = 512,
    ) -> dict[str, torch.Tensor]:
        """Tokenize a batch of texts.

        Args:
            texts: List of texts to tokenize
            max_length: Maximum sequence length

        Returns:
            Dictionary with input_ids, attention_mask tensors
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        # Move to device
        return {k: v.to(self.device) for k, v in encoded.items()}

    def _run_inference(
        self,
        batch: dict[str, torch.Tensor],
        capability: str,
    ) -> None:
        """Run single inference pass.

        Args:
            batch: Tokenized batch
            capability: Capability to use for inference
        """
        with torch.no_grad():
            # Check if model supports capability parameter
            if hasattr(self.model, "forward"):
                sig = self.model.forward.__code__.co_varnames
                if "capability" in sig:
                    self.model(**batch, capability=capability)
                else:
                    self.model(**batch)
            else:
                self.model(**batch)

    def run(
        self,
        texts: list[str],
        batch_size: int = 1,
        warmup: int = 10,
        capability: str = "sentiment",
        max_length: int = 512,
        sync_cuda: bool = True,
    ) -> LatencyResults:
        """Run latency benchmark.

        Args:
            texts: List of texts to benchmark on
            batch_size: Number of samples per inference call
            warmup: Number of warmup iterations (not measured)
            capability: Model capability to benchmark
            max_length: Maximum sequence length for tokenization
            sync_cuda: Whether to synchronize CUDA before timing

        Returns:
            LatencyResults with percentile metrics and memory usage
        """
        # Ensure model is in eval mode
        self.model.eval()

        # Prepare batches
        num_texts = len(texts)
        batches = []
        for i in range(0, num_texts, batch_size):
            batch_texts = texts[i : i + batch_size]
            batches.append(self._tokenize_batch(batch_texts, max_length))

        # Reset memory tracking
        self._reset_memory_stats()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Warmup runs
        warmup_batch = batches[0] if batches else self._tokenize_batch(texts[:1], max_length)
        for _ in range(warmup):
            self._run_inference(warmup_batch, capability)
            if sync_cuda and torch.cuda.is_available() and "cpu" not in self.device:
                torch.cuda.synchronize()

        # Reset memory after warmup
        self._reset_memory_stats()

        # Measure latencies
        latencies_ms: list[float] = []

        for batch in batches:
            # Synchronize before timing
            if sync_cuda and torch.cuda.is_available() and "cpu" not in self.device:
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            self._run_inference(batch, capability)

            # Synchronize after inference
            if sync_cuda and torch.cuda.is_available() and "cpu" not in self.device:
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            # Record latency in milliseconds
            latency_ms = (end_time - start_time) * 1000
            latencies_ms.append(latency_ms)

        # Get peak memory
        memory_mb = self._get_memory_mb()

        # Calculate statistics
        latencies_array = np.array(latencies_ms)

        p50_ms = float(np.percentile(latencies_array, 50))
        p95_ms = float(np.percentile(latencies_array, 95))
        p99_ms = float(np.percentile(latencies_array, 99))
        mean_ms = float(np.mean(latencies_array))
        std_ms = float(np.std(latencies_array))
        min_ms = float(np.min(latencies_array))
        max_ms = float(np.max(latencies_array))

        # Calculate throughput (samples per second)
        total_time_sec = sum(latencies_ms) / 1000
        throughput = num_texts / total_time_sec if total_time_sec > 0 else 0.0

        return LatencyResults(
            p50_ms=p50_ms,
            p95_ms=p95_ms,
            p99_ms=p99_ms,
            mean_ms=mean_ms,
            std_ms=std_ms,
            min_ms=min_ms,
            max_ms=max_ms,
            memory_mb=memory_mb,
            throughput=throughput,
            num_samples=num_texts,
            batch_size=batch_size,
            capability=capability,
            device=self.device,
            latencies_ms=list(latencies_ms),
        )

    def run_multi_batch(
        self,
        texts: list[str],
        batch_sizes: list[int] | None = None,
        warmup: int = 10,
        capability: str = "sentiment",
        max_length: int = 512,
    ) -> dict[int, LatencyResults]:
        """Run benchmark with multiple batch sizes.

        Args:
            texts: List of texts to benchmark on
            batch_sizes: List of batch sizes to test
            warmup: Number of warmup iterations per batch size
            capability: Model capability to benchmark
            max_length: Maximum sequence length

        Returns:
            Dictionary mapping batch_size -> LatencyResults
        """
        if batch_sizes is None:
            batch_sizes = [1, 8, 16, 32]
        results = {}
        for batch_size in batch_sizes:
            results[batch_size] = self.run(
                texts=texts,
                batch_size=batch_size,
                warmup=warmup,
                capability=capability,
                max_length=max_length,
            )
        return results

    def compare_capabilities(
        self,
        texts: list[str],
        capabilities: list[str],
        batch_size: int = 1,
        warmup: int = 10,
        max_length: int = 512,
    ) -> dict[str, LatencyResults]:
        """Compare latency across different capabilities.

        Args:
            texts: List of texts to benchmark on
            capabilities: List of capabilities to compare
            batch_size: Batch size for inference
            warmup: Number of warmup iterations per capability
            max_length: Maximum sequence length

        Returns:
            Dictionary mapping capability -> LatencyResults
        """
        results = {}
        for capability in capabilities:
            results[capability] = self.run(
                texts=texts,
                batch_size=batch_size,
                warmup=warmup,
                capability=capability,
                max_length=max_length,
            )
        return results

    def generate_report(
        self,
        results: LatencyResults | dict[str | int, LatencyResults],
    ) -> str:
        """Generate a formatted benchmark report.

        Args:
            results: Single result or dict of results

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            "LATENCY BENCHMARK REPORT",
            "=" * 60,
            "",
        ]

        if isinstance(results, LatencyResults):
            lines.append(results.summary())
        else:
            for key, result in results.items():
                lines.append(f"--- {key} ---")
                lines.append(result.summary())
                lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# Base Benchmark Class
# =============================================================================


@dataclass
class BenchmarkResults:
    """
    Container for benchmark results.

    Attributes:
        name: Benchmark name
        metrics: Dictionary of metric name -> value
        num_samples: Number of samples evaluated
        execution_time_sec: Total execution time in seconds
        timestamp: When the benchmark was run
        metadata: Additional metadata (model name, device, etc.)
    """

    name: str
    metrics: dict[str, float]
    num_samples: int = 0
    execution_time_sec: float = 0.0
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            from datetime import datetime

            self.timestamp = datetime.now().isoformat()

    @property
    def primary_metric(self) -> float:
        """Get the primary metric for this benchmark."""
        # Default: first metric or average
        if self.metrics:
            return next(iter(self.metrics.values()))
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "metrics": self.metrics,
            "num_samples": self.num_samples,
            "execution_time_sec": self.execution_time_sec,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Benchmark: {self.name}",
            "-" * 40,
            f"Samples: {self.num_samples}",
            f"Time: {self.execution_time_sec:.2f}s",
            "",
            "Metrics:",
        ]
        for name, value in sorted(self.metrics.items()):
            if isinstance(value, float):
                lines.append(f"  {name}: {value:.4f}")
            else:
                lines.append(f"  {name}: {value}")
        return "\n".join(lines)


class BaseBenchmark:
    """
    Abstract base class for benchmarks.

    Subclasses must implement:
        - run(): Execute the benchmark and return results
        - name: Property returning the benchmark name

    Attributes:
        model: The model to benchmark
        tokenizer: Tokenizer for preprocessing
        device: Device to use for inference
        batch_size: Default batch size
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "auto",
        batch_size: int = 32,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size

        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Move model to device
        self.model.to(self.device)  # type: ignore[arg-type]
        self.model.eval()

    @property
    def name(self) -> str:
        """Return benchmark name."""
        return self.__class__.__name__

    def run(self, **kwargs) -> BenchmarkResults:
        """Execute the benchmark and return results."""
        raise NotImplementedError("Subclasses must implement run()")

    def _get_dataloader(self, dataset, batch_size: int | None = None):
        """Create a DataLoader for the dataset."""
        from torch.utils.data import DataLoader

        return DataLoader(
            dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=False,
            num_workers=0,
        )


# =============================================================================
# Benchmark Suite
# =============================================================================


class BenchmarkSuite:
    """
    Orchestrate multiple benchmarks and aggregate results.

    Provides:
        - Add/remove benchmarks dynamically
        - Run all benchmarks with progress tracking
        - Parallel execution support (optional)
        - Result aggregation and comparison
        - Report generation in multiple formats

    Args:
        model: The model to benchmark
        tokenizer: Tokenizer for preprocessing
        device: Device to use for inference ("cuda", "cpu", or "auto")
        output_dir: Directory to save benchmark results

    Example:
        >>> suite = BenchmarkSuite(model=model, tokenizer=tokenizer)
        >>> suite.add_benchmark("latency", LatencyBenchmark(...))
        >>> suite.add_benchmark("glue", GLUEBenchmark(...))
        >>> results = suite.run_all()
        >>> print(suite.generate_report())
    """

    def __init__(
        self,
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        device: str = "auto",
        output_dir: str = "./benchmark_results",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir

        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Benchmarks registry
        self._benchmarks: dict[str, BaseBenchmark | LatencyBenchmark] = {}
        self._results: dict[str, BenchmarkResults | LatencyResults] = {}

        # Metadata
        self._model_name = getattr(model, "name_or_path", "unknown") if model else "unknown"
        self._run_timestamp: str = ""

    def add_benchmark(
        self,
        name: str,
        benchmark: BaseBenchmark | LatencyBenchmark,
    ) -> BenchmarkSuite:
        """
        Add a benchmark to the suite.

        Args:
            name: Unique name for the benchmark
            benchmark: Benchmark instance to add

        Returns:
            Self for method chaining
        """
        self._benchmarks[name] = benchmark
        return self

    def remove_benchmark(self, name: str) -> BenchmarkSuite:
        """
        Remove a benchmark from the suite.

        Args:
            name: Name of the benchmark to remove

        Returns:
            Self for method chaining
        """
        if name in self._benchmarks:
            del self._benchmarks[name]
        return self

    def list_benchmarks(self) -> list[str]:
        """Return list of registered benchmark names."""
        return list(self._benchmarks.keys())

    def run_benchmark(
        self,
        name: str,
        **kwargs,
    ) -> BenchmarkResults | LatencyResults:
        """
        Run a specific benchmark by name.

        Args:
            name: Name of the benchmark to run
            **kwargs: Arguments to pass to the benchmark

        Returns:
            Benchmark results
        """
        if name not in self._benchmarks:
            raise ValueError(
                f"Benchmark '{name}' not found. Available: {list(self._benchmarks.keys())}"
            )

        benchmark = self._benchmarks[name]
        result = benchmark.run(**kwargs)
        self._results[name] = result
        return result

    def run_all(
        self,
        parallel: bool = False,
        progress: bool = True,
        **kwargs,
    ) -> dict[str, BenchmarkResults | LatencyResults]:
        """
        Run all registered benchmarks.

        Args:
            parallel: Whether to run benchmarks in parallel (requires concurrent.futures)
            progress: Whether to show progress bar
            **kwargs: Arguments to pass to all benchmarks

        Returns:
            Dictionary of benchmark name -> results
        """
        from datetime import datetime

        self._run_timestamp = datetime.now().isoformat()

        if parallel:
            return self._run_parallel(**kwargs)
        else:
            return self._run_sequential(progress=progress, **kwargs)

    def _run_sequential(
        self,
        progress: bool = True,
        **kwargs,
    ) -> dict[str, BenchmarkResults | LatencyResults]:
        """Run benchmarks sequentially."""
        results = {}

        benchmark_items = list(self._benchmarks.items())

        if progress:
            try:
                from tqdm import tqdm

                benchmark_items = tqdm(benchmark_items, desc="Running benchmarks")
            except ImportError:
                pass

        for name, benchmark in benchmark_items:
            try:
                result = benchmark.run(**kwargs)
                results[name] = result
                self._results[name] = result
            except Exception as e:
                import logging

                logging.getLogger(__name__).error(f"Benchmark '{name}' failed: {e}")
                # Store error as result
                results[name] = BenchmarkResults(
                    name=name,
                    metrics={"error": 1.0},
                    metadata={"error_message": str(e)},
                )

        return results

    def _run_parallel(self, **kwargs) -> dict[str, BenchmarkResults | LatencyResults]:
        """Run benchmarks in parallel using ThreadPoolExecutor."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = {}

        def run_single(name: str, benchmark) -> tuple[str, BenchmarkResults | LatencyResults]:
            try:
                return name, benchmark.run(**kwargs)
            except Exception as e:
                return name, BenchmarkResults(
                    name=name,
                    metrics={"error": 1.0},
                    metadata={"error_message": str(e)},
                )

        with ThreadPoolExecutor(max_workers=min(4, len(self._benchmarks))) as executor:
            futures = {
                executor.submit(run_single, name, benchmark): name
                for name, benchmark in self._benchmarks.items()
            }

            for future in as_completed(futures):
                name, result = future.result()
                results[name] = result
                self._results[name] = result

        return results

    def get_results(self) -> dict[str, BenchmarkResults | LatencyResults]:
        """Return all stored results from previous runs."""
        return self._results.copy()

    def get_result(self, name: str) -> BenchmarkResults | LatencyResults | None:
        """Get result for a specific benchmark."""
        return self._results.get(name)

    def aggregate_metrics(self) -> dict[str, float]:
        """
        Aggregate metrics across all benchmarks.

        Returns:
            Dictionary with aggregated metrics
        """
        if not self._results:
            return {}

        all_metrics = {}

        for name, result in self._results.items():
            if isinstance(result, LatencyResults):
                all_metrics[f"{name}_p50_ms"] = result.p50_ms
                all_metrics[f"{name}_p95_ms"] = result.p95_ms
                all_metrics[f"{name}_throughput"] = result.throughput
            elif isinstance(result, BenchmarkResults):
                for metric_name, value in result.metrics.items():
                    all_metrics[f"{name}_{metric_name}"] = value

        # Calculate averages
        numeric_values = [
            v for v in all_metrics.values() if isinstance(v, (int, float)) and v != 1.0
        ]
        if numeric_values:
            all_metrics["avg_score"] = sum(numeric_values) / len(numeric_values)

        return all_metrics

    def generate_report(self, format: str = "text") -> str:
        """
        Generate a benchmark report.

        Args:
            format: Output format ("text", "markdown", or "json")

        Returns:
            Formatted report string
        """
        if format == "json":
            return self._generate_json_report()
        elif format == "markdown":
            return self._generate_markdown_report()
        else:
            return self._generate_text_report()

    def _generate_text_report(self) -> str:
        """Generate plain text report."""
        lines = [
            "=" * 70,
            "BENCHMARK SUITE REPORT",
            "=" * 70,
            f"Model: {self._model_name}",
            f"Device: {self.device}",
            f"Timestamp: {self._run_timestamp}",
            f"Benchmarks: {len(self._results)}",
            "=" * 70,
            "",
        ]

        for name, result in self._results.items():
            lines.append(f"--- {name} ---")
            if isinstance(result, LatencyResults):
                lines.append(result.summary())
            elif isinstance(result, BenchmarkResults):
                lines.append(result.summary())
            lines.append("")

        # Aggregated metrics
        agg = self.aggregate_metrics()
        if agg:
            lines.extend(
                [
                    "=" * 70,
                    "AGGREGATED METRICS",
                    "=" * 70,
                ]
            )
            for key, value in sorted(agg.items()):
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")

        lines.append("=" * 70)
        return "\n".join(lines)

    def _generate_markdown_report(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# Benchmark Report: {self._model_name}",
            "",
            f"**Device:** {self.device}",
            f"**Timestamp:** {self._run_timestamp}",
            "",
            "## Results Summary",
            "",
            "| Benchmark | Primary Metric | Value |",
            "|-----------|----------------|-------|",
        ]

        for name, result in self._results.items():
            if isinstance(result, LatencyResults):
                lines.append(f"| {name} | P50 Latency | {result.p50_ms:.2f}ms |")
            elif isinstance(result, BenchmarkResults):
                primary = result.primary_metric
                lines.append(f"| {name} | Primary | {primary:.4f} |")

        # Detailed results
        lines.extend(
            [
                "",
                "## Detailed Results",
                "",
            ]
        )

        for name, result in self._results.items():
            lines.append(f"### {name}")
            lines.append("")
            if isinstance(result, LatencyResults):
                lines.append(f"- P50: {result.p50_ms:.2f}ms")
                lines.append(f"- P95: {result.p95_ms:.2f}ms")
                lines.append(f"- P99: {result.p99_ms:.2f}ms")
                lines.append(f"- Throughput: {result.throughput:.1f} samples/sec")
            elif isinstance(result, BenchmarkResults):
                for metric_name, value in sorted(result.metrics.items()):
                    if isinstance(value, float):
                        lines.append(f"- {metric_name}: {value:.4f}")
            lines.append("")

        return "\n".join(lines)

    def _generate_json_report(self) -> str:
        """Generate JSON report."""
        import json

        report = {
            "model": self._model_name,
            "device": self.device,
            "timestamp": self._run_timestamp,
            "benchmarks": {},
            "aggregated": self.aggregate_metrics(),
        }

        for name, result in self._results.items():
            if isinstance(result, LatencyResults):
                report["benchmarks"][name] = result.to_dict()
            elif isinstance(result, BenchmarkResults):
                report["benchmarks"][name] = result.to_dict()

        return json.dumps(report, indent=2)

    def save_results(self, path: str | None = None) -> str:
        """
        Save results to file.

        Args:
            path: Output file path. If None, uses output_dir with timestamp.

        Returns:
            Path to saved file
        """
        from pathlib import Path

        if path is None:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"{self.output_dir}/benchmark_{timestamp}.json"

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path_obj, "w") as f:
            f.write(self._generate_json_report())

        return str(path_obj)

    def compare_with_baseline(
        self,
        baseline_results: dict[str, BenchmarkResults | LatencyResults],
    ) -> dict[str, dict[str, float]]:
        """
        Compare current results with baseline.

        Args:
            baseline_results: Baseline results to compare against

        Returns:
            Dictionary with comparison metrics (improvement/regression per benchmark)
        """
        comparison = {}

        for name, current in self._results.items():
            if name not in baseline_results:
                continue

            baseline = baseline_results[name]
            comparison[name] = {}

            if isinstance(current, LatencyResults) and isinstance(baseline, LatencyResults):
                # Lower latency is better
                comparison[name]["p50_improvement"] = (
                    baseline.p50_ms - current.p50_ms
                ) / baseline.p50_ms
                comparison[name]["throughput_improvement"] = (
                    current.throughput - baseline.throughput
                ) / baseline.throughput
            elif isinstance(current, BenchmarkResults) and isinstance(baseline, BenchmarkResults):
                # Higher metrics are typically better
                for metric_name in current.metrics:
                    if metric_name in baseline.metrics:
                        baseline_val = baseline.metrics[metric_name]
                        current_val = current.metrics[metric_name]
                        if baseline_val != 0:
                            comparison[name][f"{metric_name}_improvement"] = (
                                current_val - baseline_val
                            ) / baseline_val

        return comparison


# =============================================================================
# GLUE Benchmark
# =============================================================================


# GLUE task configurations
GLUE_TASKS = {
    "sst2": {
        "dataset": "glue",
        "subset": "sst2",
        "metric": "accuracy",
        "text_cols": ["sentence"],
        "label_col": "label",
        "num_labels": 2,
        "capability": "sentiment",
    },
    "cola": {
        "dataset": "glue",
        "subset": "cola",
        "metric": "matthews_correlation",
        "text_cols": ["sentence"],
        "label_col": "label",
        "num_labels": 2,
        "capability": "sentiment",  # acceptability as binary
    },
    "mrpc": {
        "dataset": "glue",
        "subset": "mrpc",
        "metric": "f1",
        "text_cols": ["sentence1", "sentence2"],
        "label_col": "label",
        "num_labels": 2,
        "capability": "nli",  # paraphrase detection
    },
    "qqp": {
        "dataset": "glue",
        "subset": "qqp",
        "metric": "f1",
        "text_cols": ["question1", "question2"],
        "label_col": "label",
        "num_labels": 2,
        "capability": "nli",  # duplicate question detection
    },
    "stsb": {
        "dataset": "glue",
        "subset": "stsb",
        "metric": "spearman",
        "text_cols": ["sentence1", "sentence2"],
        "label_col": "label",
        "num_labels": 1,  # regression
        "capability": "embedding",
    },
    "mnli": {
        "dataset": "glue",
        "subset": "mnli",
        "metric": "accuracy",
        "text_cols": ["premise", "hypothesis"],
        "label_col": "label",
        "num_labels": 3,
        "capability": "nli",
    },
    "qnli": {
        "dataset": "glue",
        "subset": "qnli",
        "metric": "accuracy",
        "text_cols": ["question", "sentence"],
        "label_col": "label",
        "num_labels": 2,
        "capability": "nli",
    },
    "rte": {
        "dataset": "glue",
        "subset": "rte",
        "metric": "accuracy",
        "text_cols": ["sentence1", "sentence2"],
        "label_col": "label",
        "num_labels": 2,
        "capability": "nli",
    },
    "wnli": {
        "dataset": "glue",
        "subset": "wnli",
        "metric": "accuracy",
        "text_cols": ["sentence1", "sentence2"],
        "label_col": "label",
        "num_labels": 2,
        "capability": "nli",
    },
}


class GLUEBenchmark(BaseBenchmark):
    """
    GLUE Benchmark for evaluating NLU capabilities.

    Supports all 9 GLUE tasks:
        - SST-2: Sentiment classification (accuracy)
        - CoLA: Linguistic acceptability (Matthews correlation)
        - MRPC: Paraphrase detection (F1)
        - QQP: Question pair matching (F1)
        - STS-B: Semantic textual similarity (Spearman correlation)
        - MNLI: Natural language inference (accuracy)
        - QNLI: Question NLI (accuracy)
        - RTE: Recognizing textual entailment (accuracy)
        - WNLI: Winograd NLI (accuracy)

    Args:
        model: The model to benchmark
        tokenizer: Tokenizer for preprocessing
        tasks: List of GLUE task names to run (default: all)
        device: Device to use for inference
        batch_size: Batch size for evaluation
        max_length: Maximum sequence length
        split: Dataset split to use (default: "validation")

    Example:
        >>> benchmark = GLUEBenchmark(model=model, tokenizer=tokenizer, tasks=["sst2", "mnli"])
        >>> results = benchmark.run()
        >>> print(results.metrics)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        tasks: list[str] | None = None,
        device: str = "auto",
        batch_size: int = 32,
        max_length: int = 512,
        split: str = "validation",
    ):
        super().__init__(model, tokenizer, device, batch_size)
        self.max_length = max_length
        self.split = split

        # Validate and set tasks
        if tasks is None:
            self.tasks = list(GLUE_TASKS.keys())
        else:
            invalid = [t for t in tasks if t not in GLUE_TASKS]
            if invalid:
                raise ValueError(f"Invalid GLUE tasks: {invalid}. Valid: {list(GLUE_TASKS.keys())}")
            self.tasks = tasks

    @property
    def name(self) -> str:
        return "GLUEBenchmark"

    def run(self, **kwargs) -> BenchmarkResults:
        """
        Run GLUE benchmark on all specified tasks.

        Returns:
            BenchmarkResults with metrics for each task
        """
        import logging

        logger = logging.getLogger(__name__)

        start_time = time.time()
        metrics = {}
        total_samples = 0

        for task_name in self.tasks:
            try:
                task_metrics, num_samples = self._evaluate_task(task_name)
                metrics.update(task_metrics)
                total_samples += num_samples
                logger.info(f"GLUE {task_name}: {task_metrics}")
            except Exception as e:
                logger.error(f"Failed to evaluate {task_name}: {e}")
                metrics[f"{task_name}_error"] = 1.0

        # Calculate GLUE average (excluding STS-B which uses different scale)
        score_tasks = [t for t in self.tasks if t != "stsb"]
        if score_tasks:
            scores = [
                metrics.get(
                    f"{t}_{'accuracy' if t not in ['mrpc', 'qqp', 'cola'] else GLUE_TASKS[t]['metric']}",
                    0.0,
                )
                for t in score_tasks
            ]
            # Filter out zeros/errors
            valid_scores = [s for s in scores if s > 0]
            if valid_scores:
                metrics["glue_avg"] = sum(valid_scores) / len(valid_scores)

        execution_time = time.time() - start_time

        return BenchmarkResults(
            name=self.name,
            metrics=metrics,
            num_samples=total_samples,
            execution_time_sec=execution_time,
            metadata={
                "tasks": self.tasks,
                "split": self.split,
                "batch_size": self.batch_size,
                "max_length": self.max_length,
            },
        )

    def _evaluate_task(self, task_name: str) -> tuple[dict[str, float], int]:
        """Evaluate a single GLUE task."""
        from datasets import load_dataset
        from scipy.stats import pearsonr, spearmanr
        from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

        task_config = GLUE_TASKS[task_name]

        # Load dataset
        split = self.split
        if task_name == "mnli":
            split = "validation_matched"  # MNLI has matched/mismatched

        try:
            dataset = load_dataset(task_config["dataset"], task_config["subset"], split=split)
        except Exception as e:
            raise RuntimeError(f"Failed to load {task_name} dataset: {e}") from e

        # Filter invalid labels (e.g., -1 in some datasets)
        if task_config["label_col"] in dataset.column_names:  # type: ignore[union-attr]
            dataset = dataset.filter(lambda x: x[task_config["label_col"]] >= 0)  # type: ignore[union-attr]

        num_samples = len(dataset)  # type: ignore[arg-type]
        if num_samples == 0:
            return {f"{task_name}_error": 1.0}, 0

        # Run inference
        predictions = []
        labels = []

        text_cols = task_config["text_cols"]
        label_col = task_config["label_col"]

        # Process in batches
        for i in range(0, num_samples, self.batch_size):
            batch = dataset[i : i + self.batch_size]  # type: ignore[index]

            # Tokenize
            if len(text_cols) == 1:
                texts = batch[text_cols[0]]
                encodings = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
            else:
                texts1 = batch[text_cols[0]]
                texts2 = batch[text_cols[1]]
                encodings = self.tokenizer(
                    texts1,
                    texts2,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )

            # Move to device
            encodings = {k: v.to(self.device) for k, v in encodings.items()}

            # Inference
            with torch.no_grad():
                # Check if model has capability-based inference
                if hasattr(self.model, "forward"):
                    capability = task_config.get("capability", "sentiment")
                    try:
                        outputs = self.model(
                            input_ids=encodings["input_ids"],
                            attention_mask=encodings["attention_mask"],
                            capability=capability,
                        )
                        if hasattr(outputs, "logits"):
                            logits = outputs.logits
                        elif isinstance(outputs, dict) and "logits" in outputs:
                            logits = outputs["logits"]
                        else:
                            logits = outputs
                    except TypeError:
                        # Fallback for models without capability parameter
                        outputs = self.model(
                            input_ids=encodings["input_ids"],
                            attention_mask=encodings["attention_mask"],
                        )
                        logits = outputs.logits if hasattr(outputs, "logits") else outputs

                if task_config["num_labels"] == 1:
                    # Regression (STS-B)
                    preds = logits.squeeze(-1).cpu().numpy()
                else:
                    # Classification
                    preds = logits.argmax(dim=-1).cpu().numpy()

                predictions.extend(preds.tolist())
                labels.extend(batch[label_col])

        # Calculate metrics
        predictions = np.array(predictions)
        labels = np.array(labels)

        metrics = {}
        metric_type = task_config["metric"]

        if metric_type == "accuracy":
            metrics[f"{task_name}_accuracy"] = accuracy_score(labels, predictions)
        elif metric_type == "f1":
            metrics[f"{task_name}_f1"] = f1_score(labels, predictions, average="binary")
            metrics[f"{task_name}_accuracy"] = accuracy_score(labels, predictions)
        elif metric_type == "matthews_correlation":
            metrics[f"{task_name}_matthews_correlation"] = matthews_corrcoef(
                labels, predictions.round()
            )
            metrics[f"{task_name}_accuracy"] = accuracy_score(labels, predictions.round())
        elif metric_type == "spearman":
            # STS-B: scale predictions to 0-5 range if needed
            if predictions.max() <= 1.0:
                predictions = predictions * 5.0
            spearman_corr, _ = spearmanr(labels, predictions)
            pearson_corr, _ = pearsonr(labels, predictions)
            metrics[f"{task_name}_spearman"] = spearman_corr
            metrics[f"{task_name}_pearson"] = pearson_corr

        return metrics, num_samples

    def run_subset(self, tasks: list[str], **kwargs) -> BenchmarkResults:
        """
        Run benchmark on a subset of tasks.

        Args:
            tasks: List of GLUE task names to run

        Returns:
            BenchmarkResults for the specified tasks
        """
        original_tasks = self.tasks
        self.tasks = [t for t in tasks if t in GLUE_TASKS]
        result = self.run(**kwargs)
        self.tasks = original_tasks
        return result

    @staticmethod
    def get_available_tasks() -> list[str]:
        """Return list of available GLUE tasks."""
        return list(GLUE_TASKS.keys())


# =============================================================================
# NER Benchmark
# =============================================================================


# NER dataset configurations
NER_DATASETS = {
    "conll2003": {
        "dataset": "conll2003",
        "subset": None,
        "split": "test",
        "tokens_col": "tokens",
        "tags_col": "ner_tags",
        "capability": "ner_general",
        "label_names": [
            "O",
            "B-PER",
            "I-PER",
            "B-ORG",
            "I-ORG",
            "B-LOC",
            "I-LOC",
            "B-MISC",
            "I-MISC",
        ],
    },
    "ontonotes": {
        "dataset": "tner/ontonotes5",
        "subset": None,
        "split": "test",
        "tokens_col": "tokens",
        "tags_col": "tags",
        "capability": "ner_general",
        "label_names": None,  # Will be loaded from dataset
    },
    "familyos_ner": {
        "dataset": "local",
        "path": "data/familyos/ner_family/test.jsonl",
        "tokens_col": "tokens",
        "tags_col": "ner_tags",
        "capability": "ner_family",
        "label_names": None,  # From labels.py
    },
}


class NERBenchmark(BaseBenchmark):
    """
    NER Benchmark for evaluating named entity recognition.

    Supports:
        - CoNLL-2003: Standard NER benchmark (PER, ORG, LOC, MISC)
        - OntoNotes 5.0: More fine-grained entities (18 types)
        - FamilyOS NER: Family-specific entities (21 BIO tags)

    Uses seqeval for entity-level F1, precision, recall evaluation.

    Args:
        model: The model to benchmark
        tokenizer: Tokenizer for preprocessing
        datasets: List of NER dataset names to evaluate (default: all)
        device: Device to use for inference
        batch_size: Batch size for evaluation
        max_length: Maximum sequence length
        label_list: Optional override for label list

    Example:
        >>> benchmark = NERBenchmark(model=model, tokenizer=tokenizer)
        >>> results = benchmark.run(datasets=["conll2003"])
        >>> print(results.metrics["conll2003_f1"])
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        datasets: list[str] | None = None,
        device: str = "auto",
        batch_size: int = 16,
        max_length: int = 512,
        label_list: dict[str, list[str]] | None = None,
    ):
        super().__init__(model, tokenizer, device, batch_size)
        self.max_length = max_length
        self.label_list = label_list or {}

        # Validate and set datasets
        if datasets is None:
            self.datasets = ["conll2003"]  # Default to CoNLL only
        else:
            invalid = [d for d in datasets if d not in NER_DATASETS]
            if invalid:
                raise ValueError(
                    f"Invalid NER datasets: {invalid}. Valid: {list(NER_DATASETS.keys())}"
                )
            self.datasets = datasets

    @property
    def name(self) -> str:
        return "NERBenchmark"

    def run(self, **kwargs) -> BenchmarkResults:
        """
        Run NER benchmark on all specified datasets.

        Returns:
            BenchmarkResults with metrics for each dataset
        """
        import logging

        logger = logging.getLogger(__name__)

        start_time = time.time()
        metrics = {}
        total_samples = 0

        for dataset_name in self.datasets:
            try:
                dataset_metrics, num_samples = self._evaluate_dataset(dataset_name)
                metrics.update(dataset_metrics)
                total_samples += num_samples
                logger.info(
                    f"NER {dataset_name}: F1={dataset_metrics.get(f'{dataset_name}_f1', 0):.4f}"
                )
            except Exception as e:
                logger.error(f"Failed to evaluate {dataset_name}: {e}")
                metrics[f"{dataset_name}_error"] = 1.0

        # Calculate average F1
        f1_scores = [metrics.get(f"{d}_f1", 0.0) for d in self.datasets]
        valid_f1 = [s for s in f1_scores if s > 0]
        if valid_f1:
            metrics["avg_f1"] = sum(valid_f1) / len(valid_f1)

        execution_time = time.time() - start_time

        return BenchmarkResults(
            name=self.name,
            metrics=metrics,
            num_samples=total_samples,
            execution_time_sec=execution_time,
            metadata={
                "datasets": self.datasets,
                "batch_size": self.batch_size,
                "max_length": self.max_length,
            },
        )

    def _evaluate_dataset(self, dataset_name: str) -> tuple[dict[str, float], int]:
        """Evaluate a single NER dataset."""
        try:
            from seqeval.metrics import (  # type: ignore[import-not-found]
                classification_report,
                f1_score,
                precision_score,
                recall_score,
            )
        except ImportError as exc:
            raise ImportError(
                "seqeval is required for NER evaluation: pip install seqeval"
            ) from exc

        config = NER_DATASETS[dataset_name]

        # Load dataset
        dataset = self._load_ner_dataset(dataset_name, config)
        num_samples = len(dataset)  # type: ignore[arg-type]

        if num_samples == 0:
            return {f"{dataset_name}_error": 1.0}, 0

        # Get label names
        label_names = self._get_label_names(dataset_name, config, dataset)

        # Run inference
        all_predictions = []
        all_labels = []

        tokens_col = config["tokens_col"]
        tags_col = config["tags_col"]
        capability = config.get("capability", "ner_general")

        for i in range(0, num_samples, self.batch_size):
            batch_samples = dataset[i : i + self.batch_size]  # type: ignore[index]

            for j in range(len(batch_samples[tokens_col])):  # type: ignore[arg-type]
                tokens = batch_samples[tokens_col][j]  # type: ignore[index]
                true_tags = batch_samples[tags_col][j]  # type: ignore[index]

                # Tokenize
                encodings = self.tokenizer(
                    tokens,  # type: ignore[arg-type]
                    is_split_into_words=True,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )

                # Get word IDs for alignment
                word_ids = encodings.word_ids(batch_index=0)

                # Move to device
                encodings = {k: v.to(self.device) for k, v in encodings.items()}

                # Inference
                with torch.no_grad():
                    try:
                        outputs = self.model(
                            input_ids=encodings["input_ids"],
                            attention_mask=encodings["attention_mask"],
                            capability=capability,
                        )
                        if hasattr(outputs, "logits"):
                            logits = outputs.logits
                        elif isinstance(outputs, dict) and "logits" in outputs:
                            logits = outputs["logits"]
                        else:
                            logits = outputs
                    except TypeError:
                        outputs = self.model(
                            input_ids=encodings["input_ids"],
                            attention_mask=encodings["attention_mask"],
                        )
                        logits = outputs.logits if hasattr(outputs, "logits") else outputs

                    preds = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

                # Align predictions with words
                aligned_preds = self._align_predictions(preds, word_ids, len(tokens))  # type: ignore[arg-type]
                aligned_labels = true_tags[: len(aligned_preds)]  # type: ignore[index]

                # Convert to string labels
                pred_labels = [
                    label_names[p] if p < len(label_names) else "O" for p in aligned_preds
                ]
                true_labels = [
                    label_names[label_idx] if label_idx < len(label_names) else "O"
                    for label_idx in aligned_labels
                ]

                all_predictions.append(pred_labels)
                all_labels.append(true_labels)

        # Calculate metrics
        metrics = {
            f"{dataset_name}_f1": f1_score(all_labels, all_predictions),
            f"{dataset_name}_precision": precision_score(all_labels, all_predictions),
            f"{dataset_name}_recall": recall_score(all_labels, all_predictions),
        }

        # Add per-entity metrics
        try:
            report = classification_report(all_labels, all_predictions, output_dict=True)
            for entity_type, entity_metrics in report.items():
                if isinstance(entity_metrics, dict) and entity_type not in [
                    "micro avg",
                    "macro avg",
                    "weighted avg",
                ]:
                    metrics[f"{dataset_name}_{entity_type}_f1"] = entity_metrics.get(
                        "f1-score", 0.0
                    )
        except Exception:
            pass

        return metrics, num_samples

    def _load_ner_dataset(self, dataset_name: str, config: dict):
        """Load NER dataset from HuggingFace or local file."""
        if config.get("dataset") == "local":
            # Load from local JSONL
            import json
            from pathlib import Path

            path = Path(config["path"])
            if not path.exists():
                raise FileNotFoundError(f"Local NER dataset not found: {path}")

            data = []
            with open(path) as f:
                for line in f:
                    data.append(json.loads(line))

            # Convert to dict format
            from datasets import Dataset

            return Dataset.from_list(data)
        else:
            # Load from HuggingFace
            from datasets import load_dataset

            if config.get("subset"):
                dataset = load_dataset(
                    config["dataset"], config["subset"], split=config.get("split", "test")
                )
            else:
                dataset = load_dataset(config["dataset"], split=config.get("split", "test"))

            return dataset

    def _get_label_names(self, dataset_name: str, config: dict, dataset) -> list[str]:
        """Get label names for the dataset."""
        # Check if provided in label_list
        if dataset_name in self.label_list:
            return self.label_list[dataset_name]

        # Check config
        if config.get("label_names"):
            return config["label_names"]

        # Try to get from dataset features
        try:
            tags_col = config["tags_col"]
            if tags_col in dataset.features:
                feature = dataset.features[tags_col]
                if hasattr(feature, "feature") and hasattr(feature.feature, "names"):
                    return feature.feature.names
        except Exception:
            pass

        # Fallback: try to load from labels.py
        try:
            from modeling_studio.data.labels import NER_FAMILY_LABELS, NER_GENERAL_LABELS

            capability = config.get("capability", "ner_general")
            if capability == "ner_family":
                return list(NER_FAMILY_LABELS.label2id.keys())
            else:
                return list(NER_GENERAL_LABELS.label2id.keys())
        except ImportError:
            pass

        # Ultimate fallback: generic BIO
        return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

    def _align_predictions(
        self,
        predictions: np.ndarray,
        word_ids: list[int | None],
        num_words: int,
    ) -> list[int]:
        """Align subword predictions back to word-level."""
        aligned = []
        previous_word_id = None

        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id != previous_word_id:
                # First token of a word
                aligned.append(int(predictions[idx]))
            previous_word_id = word_id

        # Pad or truncate to match original length
        if len(aligned) < num_words:
            aligned.extend([0] * (num_words - len(aligned)))
        elif len(aligned) > num_words:
            aligned = aligned[:num_words]

        return aligned

    @staticmethod
    def get_available_datasets() -> list[str]:
        """Return list of available NER datasets."""
        return list(NER_DATASETS.keys())


# =============================================================================
# Embedding Benchmark
# =============================================================================


# Embedding dataset configurations
EMBEDDING_DATASETS = {
    "stsb": {
        "dataset": "glue",
        "subset": "stsb",
        "split": "validation",
        "sentence1_col": "sentence1",
        "sentence2_col": "sentence2",
        "score_col": "label",
        "score_range": (0, 5),  # Original STS-B range
    },
    "sts12": {
        "dataset": "mteb/sts12-sts",
        "subset": None,
        "split": "test",
        "sentence1_col": "sentence1",
        "sentence2_col": "sentence2",
        "score_col": "score",
        "score_range": (0, 5),
    },
    "sts13": {
        "dataset": "mteb/sts13-sts",
        "subset": None,
        "split": "test",
        "sentence1_col": "sentence1",
        "sentence2_col": "sentence2",
        "score_col": "score",
        "score_range": (0, 5),
    },
    "sts14": {
        "dataset": "mteb/sts14-sts",
        "subset": None,
        "split": "test",
        "sentence1_col": "sentence1",
        "sentence2_col": "sentence2",
        "score_col": "score",
        "score_range": (0, 5),
    },
    "sts15": {
        "dataset": "mteb/sts15-sts",
        "subset": None,
        "split": "test",
        "sentence1_col": "sentence1",
        "sentence2_col": "sentence2",
        "score_col": "score",
        "score_range": (0, 5),
    },
    "sts16": {
        "dataset": "mteb/sts16-sts",
        "subset": None,
        "split": "test",
        "sentence1_col": "sentence1",
        "sentence2_col": "sentence2",
        "score_col": "score",
        "score_range": (0, 5),
    },
    "sick_r": {
        "dataset": "mteb/sickr-sts",
        "subset": None,
        "split": "test",
        "sentence1_col": "sentence1",
        "sentence2_col": "sentence2",
        "score_col": "score",
        "score_range": (1, 5),
    },
}


class EmbeddingBenchmark(BaseBenchmark):
    """
    Embedding Benchmark for evaluating sentence embeddings.

    Evaluates semantic textual similarity using Spearman and Pearson correlation.

    Supports:
        - STS-B: Semantic Textual Similarity Benchmark
        - STS12-16: SemEval STS tasks (2012-2016)
        - SICK-R: Sentences Involving Compositional Knowledge (relatedness)

    Args:
        model: The model to benchmark
        tokenizer: Tokenizer for preprocessing
        datasets: List of embedding dataset names to evaluate (default: ["stsb"])
        device: Device to use for inference
        batch_size: Batch size for evaluation
        max_length: Maximum sequence length
        pooling: Pooling strategy ("cls", "mean", "max")

    Example:
        >>> benchmark = EmbeddingBenchmark(model=model, tokenizer=tokenizer)
        >>> results = benchmark.run(datasets=["stsb", "sick_r"])
        >>> print(results.metrics["stsb_spearman"])
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        datasets: list[str] | None = None,
        device: str = "auto",
        batch_size: int = 32,
        max_length: int = 512,
        pooling: str = "mean",
    ):
        super().__init__(model, tokenizer, device, batch_size)
        self.max_length = max_length
        self.pooling = pooling

        # Validate and set datasets
        if datasets is None:
            self.datasets = ["stsb"]  # Default to STS-B only
        else:
            invalid = [d for d in datasets if d not in EMBEDDING_DATASETS]
            if invalid:
                raise ValueError(
                    f"Invalid embedding datasets: {invalid}. Valid: {list(EMBEDDING_DATASETS.keys())}"
                )
            self.datasets = datasets

    @property
    def name(self) -> str:
        return "EmbeddingBenchmark"

    def run(self, **kwargs) -> BenchmarkResults:
        """
        Run embedding benchmark on all specified datasets.

        Returns:
            BenchmarkResults with Spearman/Pearson correlations for each dataset
        """
        import logging

        logger = logging.getLogger(__name__)

        start_time = time.time()
        metrics = {}
        total_samples = 0

        for dataset_name in self.datasets:
            try:
                dataset_metrics, num_samples = self._evaluate_dataset(dataset_name)
                metrics.update(dataset_metrics)
                total_samples += num_samples
                logger.info(
                    f"Embedding {dataset_name}: Spearman={dataset_metrics.get(f'{dataset_name}_spearman', 0):.4f}"
                )
            except Exception as e:
                logger.error(f"Failed to evaluate {dataset_name}: {e}")
                metrics[f"{dataset_name}_error"] = 1.0

        # Calculate average Spearman
        spearman_scores = [metrics.get(f"{d}_spearman", 0.0) for d in self.datasets]
        valid_scores = [s for s in spearman_scores if s > 0]
        if valid_scores:
            metrics["avg_spearman"] = sum(valid_scores) / len(valid_scores)

        execution_time = time.time() - start_time

        return BenchmarkResults(
            name=self.name,
            metrics=metrics,
            num_samples=total_samples,
            execution_time_sec=execution_time,
            metadata={
                "datasets": self.datasets,
                "batch_size": self.batch_size,
                "max_length": self.max_length,
                "pooling": self.pooling,
            },
        )

    def _evaluate_dataset(self, dataset_name: str) -> tuple[dict[str, float], int]:
        """Evaluate a single embedding dataset."""
        from scipy.stats import pearsonr, spearmanr

        config = EMBEDDING_DATASETS[dataset_name]

        # Load dataset
        dataset = self._load_embedding_dataset(dataset_name, config)
        num_samples = len(dataset)  # type: ignore[arg-type]

        if num_samples == 0:
            return {f"{dataset_name}_error": 1.0}, 0

        # Get embeddings for both sentence columns
        sentence1_col = config["sentence1_col"]
        sentence2_col = config["sentence2_col"]
        score_col = config["score_col"]

        embeddings1 = []
        embeddings2 = []
        gold_scores = []

        for i in range(0, num_samples, self.batch_size):
            batch = dataset[i : i + self.batch_size]  # type: ignore[index]

            # Encode sentence1
            emb1 = self._encode_sentences(list(batch[sentence1_col]))  # type: ignore[arg-type]
            embeddings1.extend(emb1)

            # Encode sentence2
            emb2 = self._encode_sentences(list(batch[sentence2_col]))  # type: ignore[arg-type]
            embeddings2.extend(emb2)

            # Gold scores
            gold_scores.extend(batch[score_col])

        # Convert to numpy
        embeddings1 = np.array(embeddings1)
        embeddings2 = np.array(embeddings2)
        gold_scores = np.array(gold_scores)

        # Calculate cosine similarity
        cosine_similarities = self._cosine_similarity(embeddings1, embeddings2)

        # Calculate correlations
        spearman_result = spearmanr(gold_scores, cosine_similarities)
        pearson_result = pearsonr(gold_scores, cosine_similarities)

        # Extract correlation values (scipy returns named tuple with .statistic or index 0)
        spearman_corr: float = float(getattr(spearman_result, "statistic", spearman_result[0]))  # type: ignore[arg-type]
        pearson_corr: float = float(getattr(pearson_result, "statistic", pearson_result[0]))  # type: ignore[arg-type]

        metrics: dict[str, float] = {
            f"{dataset_name}_spearman": spearman_corr,
            f"{dataset_name}_pearson": pearson_corr,
        }

        return metrics, num_samples

    def _load_embedding_dataset(self, dataset_name: str, config: dict):
        """Load embedding dataset from HuggingFace."""
        from datasets import load_dataset

        try:
            if config.get("subset"):
                dataset = load_dataset(
                    config["dataset"], config["subset"], split=config.get("split", "test")
                )
            else:
                dataset = load_dataset(config["dataset"], split=config.get("split", "test"))
        except Exception:
            # Fallback for some datasets that may have different names
            if config["dataset"].startswith("mteb/"):
                # Try without mteb prefix
                alt_name = config["dataset"].replace("mteb/", "")
                dataset = load_dataset(alt_name, split=config.get("split", "test"))
            else:
                raise

        # Filter out invalid scores
        score_col = config["score_col"]
        if score_col in dataset.column_names:
            dataset = dataset.filter(lambda x: x[score_col] is not None)

        return dataset

    def _encode_sentences(self, sentences: list[str]) -> list[np.ndarray]:
        """Encode a batch of sentences to embeddings."""
        # Tokenize
        encodings = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Move to device
        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        # Get embeddings
        with torch.no_grad():
            try:
                # Try capability-based inference
                outputs = self.model(
                    input_ids=encodings["input_ids"],
                    attention_mask=encodings["attention_mask"],
                    capability="embedding",
                )
                if hasattr(outputs, "embeddings"):
                    embeddings = outputs.embeddings
                elif hasattr(outputs, "last_hidden_state"):
                    embeddings = self._pool(outputs.last_hidden_state, encodings["attention_mask"])
                elif isinstance(outputs, dict) and "embeddings" in outputs:
                    embeddings = outputs["embeddings"]
                elif isinstance(outputs, torch.Tensor):
                    if outputs.dim() == 3:
                        embeddings = self._pool(outputs, encodings["attention_mask"])
                    else:
                        embeddings = outputs
                else:
                    embeddings = self._pool(outputs.last_hidden_state, encodings["attention_mask"])
            except (TypeError, AttributeError):
                # Fallback for standard transformer models
                outputs = self.model(
                    input_ids=encodings["input_ids"],
                    attention_mask=encodings["attention_mask"],
                    output_hidden_states=True,
                )
                if hasattr(outputs, "last_hidden_state"):
                    embeddings = self._pool(outputs.last_hidden_state, encodings["attention_mask"])
                else:
                    embeddings = self._pool(outputs[0], encodings["attention_mask"])

        return embeddings.cpu().numpy().tolist()

    def _pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply pooling strategy to hidden states."""
        if self.pooling == "cls":
            return hidden_states[:, 0]
        elif self.pooling == "max":
            # Masked max pooling
            masked = hidden_states.masked_fill(~attention_mask.unsqueeze(-1).bool(), float("-inf"))
            return masked.max(dim=1).values
        else:  # mean
            # Masked mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
            sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
            return sum_embeddings / sum_mask

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between paired embeddings."""
        # Normalize
        a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)

        # Dot product
        return np.sum(a_norm * b_norm, axis=1)

    @staticmethod
    def get_available_datasets() -> list[str]:
        """Return list of available embedding datasets."""
        return list(EMBEDDING_DATASETS.keys())


# =============================================================================
# FamilyOS Domain Benchmark
# =============================================================================


# FamilyOS capability configurations with quality targets from enhanced_design_v2.md
FAMILYOS_CAPABILITIES = {
    "ner_family": {
        "type": "token_classification",
        "data_path": "data/familyos/ner_family/test.jsonl",
        "metric": "f1",
        "target": 0.88,  # Target F1 >= 88%
        "tokens_col": "tokens",
        "tags_col": "ner_tags",
    },
    "ingress": {
        "type": "classification",
        "data_path": "data/familyos/ingress/test.jsonl",
        "metric": "accuracy",
        "target": 0.90,  # Target Accuracy >= 90%
        "text_col": "text",
        "label_col": "label",
    },
    "safety_familyos": {
        "type": "classification",
        "data_path": "data/familyos/safety/test.jsonl",
        "metric": "macro_f1",
        "target": 0.80,  # Target Macro F1 >= 80%
        "text_col": "text",
        "label_col": "label",
        "crisis_recall_target": 0.98,  # CRISIS recall >= 98%
    },
    "relation": {
        "type": "classification",
        "data_path": "data/familyos/relations/test.jsonl",
        "metric": "f1",
        "target": 0.82,  # Target F1 >= 82%
        "text_col": "text",
        "label_col": "relation",
        "entity_cols": ["entity1", "entity2"],
    },
    "intent": {
        "type": "classification",
        "data_path": "data/familyos/intents/test.jsonl",
        "metric": "accuracy",
        "target": 0.90,  # Target Accuracy >= 90%
        "text_col": "text",
        "label_col": "label",
    },
}


class FamilyOSBenchmark(BaseBenchmark):
    """
    FamilyOS Domain Benchmark for evaluating family-specific capabilities.

    Evaluates the 5 FamilyOS-specific capabilities:
        - ner_family: Family entity recognition (21 BIO tags)
        - ingress: Activity domain classification (12 domains)
        - safety_familyos: Safety policy band classification (GREEN/AMBER/RED/CRISIS)
        - relation: Family relationship extraction (15 relations)
        - intent: User intent classification (8 intents)

    Quality targets from enhanced_design_v2.md:
        - NER Family F1 >= 88%
        - Ingress Accuracy >= 90%
        - Safety Macro F1 >= 80%, CRISIS Recall >= 98%
        - Relation F1 >= 82%
        - Intent Accuracy >= 90%

    Args:
        model: The model to benchmark
        tokenizer: Tokenizer for preprocessing
        capabilities: List of FamilyOS capabilities to evaluate (default: all)
        data_dir: Base directory for FamilyOS test data
        device: Device to use for inference
        batch_size: Batch size for evaluation
        max_length: Maximum sequence length

    Example:
        >>> benchmark = FamilyOSBenchmark(model=model, tokenizer=tokenizer)
        >>> results = benchmark.run()
        >>> print(results.metrics["ner_family_f1"])
        >>> print(results.metrics["safety_familyos_crisis_recall"])
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        capabilities: list[str] | None = None,
        data_dir: str = ".",
        device: str = "auto",
        batch_size: int = 16,
        max_length: int = 512,
    ):
        super().__init__(model, tokenizer, device, batch_size)
        self.data_dir = data_dir
        self.max_length = max_length

        # Validate and set capabilities
        if capabilities is None:
            self.capabilities = list(FAMILYOS_CAPABILITIES.keys())
        else:
            invalid = [c for c in capabilities if c not in FAMILYOS_CAPABILITIES]
            if invalid:
                raise ValueError(
                    f"Invalid FamilyOS capabilities: {invalid}. Valid: {list(FAMILYOS_CAPABILITIES.keys())}"
                )
            self.capabilities = capabilities

    @property
    def name(self) -> str:
        return "FamilyOSBenchmark"

    def run(self, **kwargs) -> BenchmarkResults:
        """
        Run FamilyOS benchmark on all specified capabilities.

        Returns:
            BenchmarkResults with metrics for each capability and target comparison
        """
        import logging

        logger = logging.getLogger(__name__)

        start_time = time.time()
        metrics = {}
        total_samples = 0
        targets_met = {}

        for capability in self.capabilities:
            try:
                cap_metrics, num_samples = self._evaluate_capability(capability)
                metrics.update(cap_metrics)
                total_samples += num_samples

                # Check if targets are met
                config = FAMILYOS_CAPABILITIES[capability]
                primary_metric = f"{capability}_{config['metric']}"
                target = config["target"]
                achieved = cap_metrics.get(primary_metric, 0.0)
                targets_met[capability] = achieved >= target

                logger.info(
                    f"FamilyOS {capability}: {primary_metric}={achieved:.4f} "
                    f"(target={target:.4f}, {'✓' if targets_met[capability] else '✗'})"
                )
            except Exception as e:
                logger.error(f"Failed to evaluate {capability}: {e}")
                metrics[f"{capability}_error"] = 1.0
                targets_met[capability] = False

        # Calculate summary metrics
        primary_metrics = []
        for cap in self.capabilities:
            config = FAMILYOS_CAPABILITIES[cap]
            metric_name = f"{cap}_{config['metric']}"
            if metric_name in metrics:
                primary_metrics.append(metrics[metric_name])

        if primary_metrics:
            metrics["avg_primary_metric"] = sum(primary_metrics) / len(primary_metrics)

        metrics["targets_met_count"] = sum(targets_met.values())
        metrics["targets_total"] = len(self.capabilities)
        metrics["all_targets_met"] = 1.0 if all(targets_met.values()) else 0.0

        execution_time = time.time() - start_time

        return BenchmarkResults(
            name=self.name,
            metrics=metrics,
            num_samples=total_samples,
            execution_time_sec=execution_time,
            metadata={
                "capabilities": self.capabilities,
                "batch_size": self.batch_size,
                "max_length": self.max_length,
                "targets_met": targets_met,
                "quality_targets": {
                    cap: FAMILYOS_CAPABILITIES[cap]["target"] for cap in self.capabilities
                },
            },
        )

    def _evaluate_capability(self, capability: str) -> tuple[dict[str, float], int]:
        """Evaluate a single FamilyOS capability."""
        config = FAMILYOS_CAPABILITIES[capability]
        cap_type = config["type"]

        if cap_type == "token_classification":
            return self._evaluate_token_classification(capability, config)
        elif cap_type == "classification":
            return self._evaluate_classification(capability, config)
        else:
            raise ValueError(f"Unknown capability type: {cap_type}")

    def _evaluate_token_classification(
        self,
        capability: str,
        config: dict,
    ) -> tuple[dict[str, float], int]:
        """Evaluate token classification capability (NER)."""
        try:
            from seqeval.metrics import (  # type: ignore[import-not-found]
                f1_score,
                precision_score,
                recall_score,
            )
        except ImportError as exc:
            raise ImportError("seqeval required for NER evaluation") from exc

        # Load dataset
        dataset = self._load_dataset(config["data_path"])
        num_samples = len(dataset)

        if num_samples == 0:
            return {f"{capability}_error": 1.0}, 0

        # Get label names from labels.py
        label_names = self._get_label_names(capability)

        tokens_col = config["tokens_col"]
        tags_col = config["tags_col"]

        all_predictions = []
        all_labels = []

        for sample in dataset:
            tokens = sample[tokens_col]
            true_tags = sample[tags_col]

            # Tokenize
            encodings = self.tokenizer(
                tokens,
                is_split_into_words=True,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            word_ids = encodings.word_ids(batch_index=0)
            encodings = {k: v.to(self.device) for k, v in encodings.items()}

            # Inference
            with torch.no_grad():
                try:
                    outputs = self.model(
                        input_ids=encodings["input_ids"],
                        attention_mask=encodings["attention_mask"],
                        capability=capability,
                    )
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                except TypeError:
                    outputs = self.model(
                        input_ids=encodings["input_ids"],
                        attention_mask=encodings["attention_mask"],
                    )
                    logits = outputs.logits

                preds = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

            # Align predictions
            aligned_preds = self._align_ner_predictions(preds, word_ids, len(tokens))
            aligned_labels = true_tags[: len(aligned_preds)]

            # Convert to string labels
            pred_labels = [label_names[p] if p < len(label_names) else "O" for p in aligned_preds]
            true_labels = [
                label_names[tag_idx] if tag_idx < len(label_names) else "O"
                for tag_idx in aligned_labels
            ]

            all_predictions.append(pred_labels)
            all_labels.append(true_labels)

        metrics = {
            f"{capability}_f1": f1_score(all_labels, all_predictions),
            f"{capability}_precision": precision_score(all_labels, all_predictions),
            f"{capability}_recall": recall_score(all_labels, all_predictions),
        }

        return metrics, num_samples

    def _evaluate_classification(
        self,
        capability: str,
        config: dict,
    ) -> tuple[dict[str, float], int]:
        """Evaluate classification capability."""
        from sklearn.metrics import accuracy_score, f1_score

        # Load dataset
        dataset = self._load_dataset(config["data_path"])
        num_samples = len(dataset)

        if num_samples == 0:
            return {f"{capability}_error": 1.0}, 0

        text_col = config["text_col"]
        label_col = config["label_col"]

        all_predictions = []
        all_labels = []
        all_probs = []  # For CRISIS recall calculation

        for i in range(0, num_samples, self.batch_size):
            batch = dataset[i : i + self.batch_size]
            texts = [sample[text_col] for sample in batch]
            labels = [sample[label_col] for sample in batch]

            # Handle relation extraction with entity markers
            if capability == "relation" and "entity_cols" in config:
                # Add entity markers to text
                texts = [
                    self._add_entity_markers(
                        sample[text_col],
                        sample.get(config["entity_cols"][0], ""),
                        sample.get(config["entity_cols"][1], ""),
                    )
                    for sample in batch
                ]

            # Tokenize
            encodings = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            encodings = {k: v.to(self.device) for k, v in encodings.items()}

            # Inference
            with torch.no_grad():
                try:
                    outputs = self.model(
                        input_ids=encodings["input_ids"],
                        attention_mask=encodings["attention_mask"],
                        capability=capability,
                    )
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                except TypeError:
                    outputs = self.model(
                        input_ids=encodings["input_ids"],
                        attention_mask=encodings["attention_mask"],
                    )
                    logits = outputs.logits

                preds = logits.argmax(dim=-1).cpu().numpy()

                # Store probabilities for safety evaluation
                if capability == "safety_familyos":
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    all_probs.extend(probs.tolist())

            all_predictions.extend(preds.tolist())
            all_labels.extend(labels)

        # Convert to numpy
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        # Calculate metrics
        metrics = {
            f"{capability}_accuracy": accuracy_score(all_labels, all_predictions),
            f"{capability}_f1": f1_score(
                all_labels, all_predictions, average="weighted", zero_division=0
            ),
            f"{capability}_macro_f1": f1_score(
                all_labels, all_predictions, average="macro", zero_division=0
            ),
        }

        # Special handling for safety: calculate CRISIS recall
        if capability == "safety_familyos" and all_probs:
            crisis_metrics = self._calculate_crisis_recall(all_labels, all_predictions, all_probs)
            metrics.update(crisis_metrics)

        return metrics, num_samples

    def _calculate_crisis_recall(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        probs: list[list[float]],
    ) -> dict[str, float]:
        """Calculate CRISIS-specific metrics for safety evaluation."""
        # Assuming CRISIS is label index 3 (GREEN=0, AMBER=1, RED=2, CRISIS=3)
        crisis_index = 3
        red_index = 2

        metrics = {}

        # CRISIS recall
        crisis_mask = labels == crisis_index
        if crisis_mask.sum() > 0:
            crisis_predictions = predictions[crisis_mask]
            crisis_recall = (crisis_predictions == crisis_index).sum() / crisis_mask.sum()
            metrics["safety_familyos_crisis_recall"] = crisis_recall

            # Also count predictions that escalated to RED or CRISIS (conservative)
            escalated_predictions = (crisis_predictions >= red_index).sum() / crisis_mask.sum()
            metrics["safety_familyos_crisis_escalation_recall"] = escalated_predictions

        # RED recall
        red_mask = labels == red_index
        if red_mask.sum() > 0:
            red_predictions = predictions[red_mask]
            red_recall = (red_predictions == red_index).sum() / red_mask.sum()
            metrics["safety_familyos_red_recall"] = red_recall

        # False positive rate for GREEN (should not escalate safe content)
        green_mask = labels == 0  # GREEN
        if green_mask.sum() > 0:
            green_predictions = predictions[green_mask]
            false_positive_rate = (green_predictions > 0).sum() / green_mask.sum()
            metrics["safety_familyos_green_fpr"] = false_positive_rate

        return metrics

    def _load_dataset(self, data_path: str) -> list[dict]:
        """Load FamilyOS dataset from JSONL file."""
        import json
        from pathlib import Path

        full_path = Path(self.data_dir) / data_path
        if not full_path.exists():
            # Return empty list if file doesn't exist (will trigger error metric)
            return []

        data = []
        with open(full_path) as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        return data

    def _get_label_names(self, capability: str) -> list[str]:
        """Get label names from labels.py."""
        try:
            from modeling_studio.data.labels import (
                INGRESS_LABELS,
                INTENT_LABELS,
                NER_FAMILY_LABELS,
                RELATION_LABELS,
                SAFETY_FAMILYOS_LABELS,
            )

            label_map = {
                "ner_family": NER_FAMILY_LABELS,
                "ingress": INGRESS_LABELS,
                "safety_familyos": SAFETY_FAMILYOS_LABELS,
                "relation": RELATION_LABELS,
                "intent": INTENT_LABELS,
            }

            if capability in label_map:
                return list(label_map[capability].label2id.keys())
        except ImportError:
            pass

        # Fallback
        return ["O"]

    def _align_ner_predictions(
        self,
        predictions: np.ndarray,
        word_ids: list[int | None],
        num_words: int,
    ) -> list[int]:
        """Align subword predictions back to word-level."""
        aligned = []
        previous_word_id = None

        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id != previous_word_id:
                aligned.append(int(predictions[idx]))
            previous_word_id = word_id

        # Pad or truncate
        if len(aligned) < num_words:
            aligned.extend([0] * (num_words - len(aligned)))
        elif len(aligned) > num_words:
            aligned = aligned[:num_words]

        return aligned

    def _add_entity_markers(self, text: str, entity1: str, entity2: str) -> str:
        """Add entity markers for relation extraction."""
        # Simple marker format: [E1] entity1 [/E1] ... [E2] entity2 [/E2]
        if entity1 in text:
            text = text.replace(entity1, f"[E1] {entity1} [/E1]", 1)
        if entity2 in text:
            text = text.replace(entity2, f"[E2] {entity2} [/E2]", 1)
        return text

    def check_quality_gates(self) -> dict[str, bool]:
        """
        Check if all quality gates are passed.

        Returns:
            Dictionary with capability -> pass/fail status
        """
        # Run benchmark if not already done
        results = self.run()

        gates = {}
        for cap in self.capabilities:
            config = FAMILYOS_CAPABILITIES[cap]
            metric_name = f"{cap}_{config['metric']}"
            target = config["target"]
            achieved = results.metrics.get(metric_name, 0.0)
            gates[cap] = achieved >= target

            # Special check for CRISIS recall
            if cap == "safety_familyos":
                crisis_recall_target = config.get("crisis_recall_target", 0.98)
                crisis_recall = results.metrics.get("safety_familyos_crisis_recall", 0.0)
                gates[f"{cap}_crisis"] = crisis_recall >= crisis_recall_target

        return gates

    @staticmethod
    def get_available_capabilities() -> list[str]:
        """Return list of available FamilyOS capabilities."""
        return list(FAMILYOS_CAPABILITIES.keys())

    @staticmethod
    def get_quality_targets() -> dict[str, float]:
        """Return quality targets for each capability."""
        return {cap: FAMILYOS_CAPABILITIES[cap]["target"] for cap in FAMILYOS_CAPABILITIES}


# =============================================================================
# Baseline Comparison
# =============================================================================


@dataclass
class ComparisonResult:
    """
    Result of comparing unified model against baselines.

    Attributes:
        unified_metrics: Metrics from unified model
        baseline_metrics: Metrics from baseline models per task
        improvements: Relative improvement per task (positive = better)
        regressions: Tasks where unified model is worse
        overall_improvement: Average improvement across all tasks
    """

    unified_metrics: dict[str, float]
    baseline_metrics: dict[str, dict[str, float]]
    improvements: dict[str, float]
    regressions: list[str]
    overall_improvement: float
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            from datetime import datetime

            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "unified_metrics": self.unified_metrics,
            "baseline_metrics": self.baseline_metrics,
            "improvements": self.improvements,
            "regressions": self.regressions,
            "overall_improvement": self.overall_improvement,
            "timestamp": self.timestamp,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "BASELINE COMPARISON RESULTS",
            "=" * 60,
            f"Timestamp: {self.timestamp}",
            f"Overall Improvement: {self.overall_improvement:+.2%}",
            "",
            "Per-Task Comparison:",
            "-" * 40,
        ]

        for task, improvement in sorted(self.improvements.items()):
            status = "✓" if improvement >= 0 else "✗"
            lines.append(f"  {status} {task}: {improvement:+.2%}")

        if self.regressions:
            lines.extend(
                [
                    "",
                    "⚠️ Regressions detected:",
                ]
            )
            for task in self.regressions:
                lines.append(f"  - {task}")

        lines.append("=" * 60)
        return "\n".join(lines)


class BaselineComparison:
    """
    Compare unified multi-task model against individual specialist baselines.

    Loads baseline models, runs the same benchmarks on both, and calculates
    relative improvement/regression for each task.

    Args:
        unified_model: The unified multi-task model to evaluate
        baselines: Dict mapping task name -> baseline model (or model path)
        tokenizer: Tokenizer for the unified model
        baseline_tokenizers: Optional dict of task -> tokenizer for baselines
        device: Device to use for inference
        batch_size: Batch size for evaluation

    Example:
        >>> comparison = BaselineComparison(
        ...     unified_model=model,
        ...     baselines={"ner": "dslim/bert-base-NER", "sentiment": "distilbert-sst2"},
        ... )
        >>> results = comparison.compare(datasets={"ner": ner_test, "sentiment": sent_test})
        >>> print(results.summary())
    """

    def __init__(
        self,
        unified_model: PreTrainedModel,
        baselines: dict[str, PreTrainedModel | str],
        tokenizer: PreTrainedTokenizer | None = None,
        baseline_tokenizers: dict[str, PreTrainedTokenizer] | None = None,
        device: str = "auto",
        batch_size: int = 32,
    ):
        self.unified_model = unified_model
        self.baselines = baselines
        self.tokenizer = tokenizer
        self.baseline_tokenizers = baseline_tokenizers or {}
        self.batch_size = batch_size

        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Loaded baseline models cache
        self._loaded_baselines: dict[str, PreTrainedModel] = {}
        self._loaded_baseline_tokenizers: dict[str, PreTrainedTokenizer] = {}

    def compare(
        self,
        datasets: dict[str, Any] | None = None,
        metrics_per_task: dict[str, str] | None = None,
        **kwargs,
    ) -> ComparisonResult:
        """
        Run comparison between unified model and baselines.

        Args:
            datasets: Dict of task name -> test dataset
            metrics_per_task: Dict of task name -> primary metric name
            **kwargs: Additional arguments passed to evaluation

        Returns:
            ComparisonResult with detailed comparison metrics
        """
        import logging

        logger = logging.getLogger(__name__)

        # Default primary metrics per task
        default_metrics = {
            "ner": "f1",
            "ner_general": "f1",
            "ner_family": "f1",
            "sentiment": "accuracy",
            "emotions": "macro_f1",
            "safety_generic": "macro_f1",
            "safety_familyos": "macro_f1",
            "nli": "accuracy",
            "ingress": "accuracy",
            "intent": "accuracy",
            "relation": "f1",
            "embedding": "spearman",
        }
        metrics_per_task = metrics_per_task or default_metrics

        unified_metrics = {}
        baseline_metrics = {}
        improvements = {}
        regressions = []

        # Evaluate each task
        for task, baseline in self.baselines.items():
            try:
                # Get dataset for this task
                dataset = datasets.get(task) if datasets else None

                # Evaluate unified model on this task
                unified_score = self._evaluate_unified(task, dataset, **kwargs)
                unified_metrics[task] = unified_score

                # Evaluate baseline model
                baseline_score = self._evaluate_baseline(task, baseline, dataset, **kwargs)
                baseline_metrics[task] = {"score": baseline_score}

                # Calculate improvement
                if baseline_score > 0:
                    improvement = (unified_score - baseline_score) / baseline_score
                else:
                    improvement = 1.0 if unified_score > 0 else 0.0

                improvements[task] = improvement

                if improvement < 0:
                    regressions.append(task)

                logger.info(
                    f"Task {task}: unified={unified_score:.4f}, "
                    f"baseline={baseline_score:.4f}, improvement={improvement:+.2%}"
                )

            except Exception as e:
                logger.error(f"Failed to compare task {task}: {e}")
                improvements[task] = 0.0

        # Calculate overall improvement
        valid_improvements = [v for v in improvements.values() if v is not None]
        overall_improvement = (
            sum(valid_improvements) / len(valid_improvements) if valid_improvements else 0.0
        )

        return ComparisonResult(
            unified_metrics=unified_metrics,
            baseline_metrics=baseline_metrics,
            improvements=improvements,
            regressions=regressions,
            overall_improvement=overall_improvement,
        )

    def _evaluate_unified(
        self,
        task: str,
        dataset: Any | None = None,
        **kwargs,
    ) -> float:
        """Evaluate unified model on a specific task."""
        if dataset is None:
            # Return dummy score if no dataset provided
            return 0.0

        # Use the evaluator for proper evaluation
        try:
            from modeling_studio.evaluation.evaluator import Evaluator

            if self.tokenizer is None:
                raise ValueError("Tokenizer is required for evaluation")

            evaluator = Evaluator(
                model=self.unified_model,
                tokenizer=self.tokenizer,
                capabilities=[task],
                device=self.device,
            )

            results = evaluator.evaluate_all(
                datasets={task: dataset},
                batch_size=self.batch_size,
            )

            # Get primary metric
            task_metrics = results.per_task.get(task, {})
            return task_metrics.get("f1", task_metrics.get("accuracy", 0.0))

        except ImportError:
            # Fallback: simple inference
            if self.tokenizer is None:
                return 0.0
            return self._simple_evaluate(self.unified_model, self.tokenizer, task, dataset)

    def _evaluate_baseline(
        self,
        task: str,
        baseline: PreTrainedModel | str,
        dataset: Any | None = None,
        **kwargs,
    ) -> float:
        """Evaluate baseline model on a specific task."""
        if dataset is None:
            return 0.0

        # Load baseline model if it's a string path
        model = self._get_baseline_model(task, baseline)
        tokenizer = self._get_baseline_tokenizer(task, baseline)

        return self._simple_evaluate(model, tokenizer, task, dataset)

    def _get_baseline_model(self, task: str, baseline: PreTrainedModel | str) -> PreTrainedModel:
        """Get or load baseline model."""
        if isinstance(baseline, str):
            if task not in self._loaded_baselines:
                from transformers import (
                    AutoModelForSequenceClassification,
                    AutoModelForTokenClassification,
                )

                try:
                    # Try sequence classification first
                    model = AutoModelForSequenceClassification.from_pretrained(baseline)
                except Exception:
                    # Fall back to token classification
                    model = AutoModelForTokenClassification.from_pretrained(baseline)

                model.to(self.device)
                model.eval()
                self._loaded_baselines[task] = model

            return self._loaded_baselines[task]
        else:
            return baseline

    def _get_baseline_tokenizer(
        self, task: str, baseline: PreTrainedModel | str
    ) -> PreTrainedTokenizer:
        """Get or load baseline tokenizer."""
        if task in self.baseline_tokenizers:
            return self.baseline_tokenizers[task]

        if isinstance(baseline, str):
            if task not in self._loaded_baseline_tokenizers:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(baseline)
                self._loaded_baseline_tokenizers[task] = tokenizer

            return self._loaded_baseline_tokenizers[task]

        # Fall back to unified model tokenizer
        if self.tokenizer is None:
            raise ValueError("No tokenizer available for baseline evaluation")
        return self.tokenizer

    def _simple_evaluate(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        task: str,
        dataset: Any,
    ) -> float:
        """Simple evaluation for a model on a dataset."""
        from sklearn.metrics import accuracy_score, f1_score

        model.to(self.device)  # type: ignore[arg-type]
        model.eval()

        predictions = []
        labels = []

        # Determine columns based on task type
        text_col = "text" if "text" in dataset.column_names else "sentence"
        label_col = "label" if "label" in dataset.column_names else "labels"

        num_samples = min(len(dataset), 1000)  # Limit for speed

        for i in range(0, num_samples, self.batch_size):
            batch = dataset[i : i + self.batch_size]

            # Get texts
            if text_col in batch:
                texts = batch[text_col]
            elif "tokens" in batch:
                # Token classification
                texts = [" ".join(tokens) for tokens in batch["tokens"]]
            else:
                continue

            # Tokenize
            encodings = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encodings = {k: v.to(self.device) for k, v in encodings.items()}

            # Inference
            with torch.no_grad():
                outputs = model(**encodings)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                preds = logits.argmax(dim=-1).cpu().numpy()

            # Handle sequence vs token classification
            if preds.ndim == 1:
                predictions.extend(preds.tolist())
            else:
                predictions.extend(preds[:, 0].tolist())

            # Labels
            if label_col in batch:
                batch_labels = batch[label_col]
                if isinstance(batch_labels[0], list):
                    labels.extend([lbl[0] for lbl in batch_labels])
                else:
                    labels.extend(batch_labels)

        if not predictions or not labels:
            return 0.0

        # Calculate metrics
        try:
            acc = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, average="weighted", zero_division=0)
            return float(max(acc, f1))  # Return best metric
        except Exception:
            return 0.0

    def generate_comparison_table(self, result: ComparisonResult) -> str:
        """Generate markdown comparison table."""
        lines = [
            "# Baseline Comparison Results",
            "",
            f"**Generated:** {result.timestamp}",
            f"**Overall Improvement:** {result.overall_improvement:+.2%}",
            "",
            "## Per-Task Comparison",
            "",
            "| Task | Unified | Baseline | Improvement | Status |",
            "|------|---------|----------|-------------|--------|",
        ]

        for task in sorted(result.improvements.keys()):
            unified = result.unified_metrics.get(task, 0.0)
            baseline = result.baseline_metrics.get(task, {}).get("score", 0.0)
            improvement = result.improvements[task]
            status = "✅" if improvement >= 0 else "❌"

            lines.append(
                f"| {task} | {unified:.4f} | {baseline:.4f} | {improvement:+.2%} | {status} |"
            )

        if result.regressions:
            lines.extend(
                [
                    "",
                    "## ⚠️ Regressions",
                    "",
                ]
            )
            for task in result.regressions:
                lines.append(f"- {task}: {result.improvements[task]:+.2%}")

        return "\n".join(lines)

    @staticmethod
    def get_default_baselines() -> dict[str, str]:
        """Return default baseline model paths."""
        return {
            "ner_general": "dslim/bert-base-NER",
            "sentiment": "distilbert-base-uncased-finetuned-sst-2-english",
            "emotions": "SamLowe/roberta-base-go_emotions",
            "nli": "facebook/bart-large-mnli",
            "embedding": "sentence-transformers/all-MiniLM-L6-v2",
        }


# =============================================================================
# Benchmark Result Tracker
# =============================================================================


class BenchmarkResultTracker:
    """
    Track and store benchmark results over time.

    Provides:
        - Result storage with timestamps and model versions
        - Historical comparison across runs
        - Regression detection
        - Export to JSON/CSV formats

    Args:
        output_dir: Directory to store benchmark results
        project_name: Optional project name for organizing results

    Example:
        >>> tracker = BenchmarkResultTracker(output_dir="./benchmark_results")
        >>> tracker.log_result("latency", results, model_version="v1")
        >>> history = tracker.get_history("latency")
        >>> tracker.detect_regressions("latency", threshold=0.05)
    """

    def __init__(
        self,
        output_dir: str = "./benchmark_results",
        project_name: str = "default",
    ):
        from pathlib import Path

        self.output_dir = Path(output_dir)
        self.project_name = project_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Index file for tracking all results
        self._index_file = self.output_dir / "index.json"
        self._index = self._load_index()

    def _load_index(self) -> dict[str, list[dict]]:
        """Load the results index."""
        import json

        if self._index_file.exists():
            with open(self._index_file) as f:
                return json.load(f)
        return {}

    def _save_index(self) -> None:
        """Save the results index."""
        import json

        with open(self._index_file, "w") as f:
            json.dump(self._index, f, indent=2)

    def log_result(
        self,
        benchmark_name: str,
        result: BenchmarkResults | LatencyResults | dict[str, Any],
        model_version: str = "unknown",
        tags: list[str] | None = None,
        notes: str = "",
    ) -> str:
        """
        Log a benchmark result.

        Args:
            benchmark_name: Name of the benchmark
            result: Benchmark result object or dict
            model_version: Version identifier for the model
            tags: Optional tags for categorization
            notes: Optional notes about this run

        Returns:
            Path to the saved result file
        """
        import json
        from datetime import datetime

        timestamp = datetime.now()
        timestamp_str = timestamp.isoformat()

        # Convert result to dict
        if hasattr(result, "to_dict"):
            result_dict = result.to_dict()
        elif isinstance(result, dict):
            result_dict = result
        else:
            result_dict = {"value": str(result)}

        # Create result entry
        entry = {
            "benchmark": benchmark_name,
            "model_version": model_version,
            "timestamp": timestamp_str,
            "tags": tags or [],
            "notes": notes,
            "project": self.project_name,
            "result": result_dict,
        }

        # Save to file
        filename = f"{benchmark_name}_{model_version}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(entry, f, indent=2)

        # Update index
        if benchmark_name not in self._index:
            self._index[benchmark_name] = []

        self._index[benchmark_name].append(
            {
                "file": filename,
                "model_version": model_version,
                "timestamp": timestamp_str,
                "tags": tags or [],
            }
        )

        self._save_index()

        return str(filepath)

    def get_result(self, benchmark_name: str, model_version: str | None = None) -> dict | None:
        """
        Get a specific result.

        Args:
            benchmark_name: Name of the benchmark
            model_version: Optional model version (gets latest if not specified)

        Returns:
            Result dict or None if not found
        """
        import json

        if benchmark_name not in self._index:
            return None

        entries = self._index[benchmark_name]
        if not entries:
            return None

        # Filter by version if specified
        if model_version:
            entries = [e for e in entries if e["model_version"] == model_version]

        if not entries:
            return None

        # Get most recent
        latest = sorted(entries, key=lambda x: x["timestamp"], reverse=True)[0]
        filepath = self.output_dir / latest["file"]

        if filepath.exists():
            with open(filepath) as f:
                return json.load(f)

        return None

    def get_history(
        self,
        benchmark_name: str,
        limit: int | None = None,
        model_version: str | None = None,
        tags: list[str] | None = None,
    ) -> list[dict]:
        """
        Get historical results for a benchmark.

        Args:
            benchmark_name: Name of the benchmark
            limit: Maximum number of results to return
            model_version: Filter by model version
            tags: Filter by tags (any match)

        Returns:
            List of result dicts, sorted by timestamp (newest first)
        """
        import json

        if benchmark_name not in self._index:
            return []

        entries = self._index[benchmark_name]

        # Apply filters
        if model_version:
            entries = [e for e in entries if e["model_version"] == model_version]

        if tags:
            entries = [e for e in entries if any(t in e.get("tags", []) for t in tags)]

        # Sort by timestamp
        entries = sorted(entries, key=lambda x: x["timestamp"], reverse=True)

        # Apply limit
        if limit:
            entries = entries[:limit]

        # Load full results
        results = []
        for entry in entries:
            filepath = self.output_dir / entry["file"]
            if filepath.exists():
                with open(filepath) as f:
                    results.append(json.load(f))

        return results

    def get_latest(self, benchmark_name: str) -> dict | None:
        """Get the most recent result for a benchmark."""
        history = self.get_history(benchmark_name, limit=1)
        return history[0] if history else None

    def compare_versions(
        self,
        benchmark_name: str,
        version_a: str,
        version_b: str,
    ) -> dict[str, Any]:
        """
        Compare results between two model versions.

        Args:
            benchmark_name: Name of the benchmark
            version_a: First model version
            version_b: Second model version

        Returns:
            Comparison dict with metrics differences
        """
        result_a = self.get_result(benchmark_name, version_a)
        result_b = self.get_result(benchmark_name, version_b)

        if not result_a or not result_b:
            return {"error": "One or both versions not found"}

        comparison = {
            "version_a": version_a,
            "version_b": version_b,
            "benchmark": benchmark_name,
            "differences": {},
        }

        metrics_a = result_a.get("result", {}).get("metrics", result_a.get("result", {}))
        metrics_b = result_b.get("result", {}).get("metrics", result_b.get("result", {}))

        for key in set(metrics_a.keys()) | set(metrics_b.keys()):
            val_a = metrics_a.get(key, 0.0)
            val_b = metrics_b.get(key, 0.0)

            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                diff = val_b - val_a
                pct_change = (diff / val_a * 100) if val_a != 0 else 0.0
                comparison["differences"][key] = {
                    "version_a": val_a,
                    "version_b": val_b,
                    "difference": diff,
                    "percent_change": pct_change,
                }

        return comparison

    def detect_regressions(
        self,
        benchmark_name: str,
        threshold: float = 0.02,
        metric: str | None = None,
    ) -> list[dict]:
        """
        Detect regressions in benchmark history.

        Args:
            benchmark_name: Name of the benchmark
            threshold: Minimum relative change to consider a regression (default: 2%)
            metric: Specific metric to check (checks all if not specified)

        Returns:
            List of detected regressions with details
        """
        history = self.get_history(benchmark_name, limit=10)

        if len(history) < 2:
            return []

        regressions = []

        for i in range(len(history) - 1):
            current = history[i]
            previous = history[i + 1]

            current_metrics = current.get("result", {}).get("metrics", current.get("result", {}))
            previous_metrics = previous.get("result", {}).get("metrics", previous.get("result", {}))

            metrics_to_check = [metric] if metric else current_metrics.keys()

            for m in metrics_to_check:
                if m not in current_metrics or m not in previous_metrics:
                    continue

                curr_val = current_metrics[m]
                prev_val = previous_metrics[m]

                if not isinstance(curr_val, (int, float)) or not isinstance(prev_val, (int, float)):
                    continue

                if prev_val == 0:
                    continue

                change = (curr_val - prev_val) / prev_val

                # Regression if value decreased more than threshold
                # (assuming higher is better for most metrics)
                if change < -threshold:
                    regressions.append(
                        {
                            "metric": m,
                            "current_version": current.get("model_version"),
                            "previous_version": previous.get("model_version"),
                            "current_value": curr_val,
                            "previous_value": prev_val,
                            "change": change,
                            "timestamp": current.get("timestamp"),
                        }
                    )

        return regressions

    def export_to_csv(self, benchmark_name: str, output_path: str | None = None) -> str:
        """
        Export benchmark history to CSV.

        Args:
            benchmark_name: Name of the benchmark
            output_path: Optional output path (generates default if not specified)

        Returns:
            Path to the exported CSV file
        """
        import csv

        history = self.get_history(benchmark_name)

        if not history:
            raise ValueError(f"No history found for benchmark: {benchmark_name}")

        # Determine output path
        if output_path is None:
            output_path = str(self.output_dir / f"{benchmark_name}_history.csv")

        # Collect all metric keys
        all_metrics = set()
        for entry in history:
            metrics = entry.get("result", {}).get("metrics", entry.get("result", {}))
            all_metrics.update(metrics.keys())

        # Sort metrics for consistent column order
        metric_cols = sorted(all_metrics)

        # Write CSV
        with open(output_path, "w", newline="") as f:
            fieldnames = ["timestamp", "model_version", "tags", "notes"] + metric_cols
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for entry in history:
                row = {
                    "timestamp": entry.get("timestamp", ""),
                    "model_version": entry.get("model_version", ""),
                    "tags": ",".join(entry.get("tags", [])),
                    "notes": entry.get("notes", ""),
                }

                metrics = entry.get("result", {}).get("metrics", entry.get("result", {}))
                for m in metric_cols:
                    row[m] = metrics.get(m, "")

                writer.writerow(row)

        return output_path

    def export_to_json(self, benchmark_name: str, output_path: str | None = None) -> str:
        """
        Export benchmark history to JSON.

        Args:
            benchmark_name: Name of the benchmark
            output_path: Optional output path

        Returns:
            Path to the exported JSON file
        """
        import json

        history = self.get_history(benchmark_name)

        if output_path is None:
            output_path = str(self.output_dir / f"{benchmark_name}_history.json")

        with open(output_path, "w") as f:
            json.dump(history, f, indent=2)

        return output_path

    def list_benchmarks(self) -> list[str]:
        """List all tracked benchmarks."""
        return list(self._index.keys())

    def list_versions(self, benchmark_name: str) -> list[str]:
        """List all model versions for a benchmark."""
        if benchmark_name not in self._index:
            return []

        versions = set()
        for entry in self._index[benchmark_name]:
            versions.add(entry["model_version"])

        return sorted(versions)

    def summary(self) -> str:
        """Generate a summary of all tracked benchmarks."""
        lines = [
            "=" * 60,
            "BENCHMARK RESULT TRACKER SUMMARY",
            "=" * 60,
            f"Project: {self.project_name}",
            f"Output Directory: {self.output_dir}",
            "",
            "Tracked Benchmarks:",
            "-" * 40,
        ]

        for benchmark in sorted(self._index.keys()):
            entries = self._index[benchmark]
            versions = len({e["model_version"] for e in entries})
            latest = sorted(entries, key=lambda x: x["timestamp"], reverse=True)[0]
            lines.append(f"  {benchmark}:")
            lines.append(f"    Runs: {len(entries)}, Versions: {versions}")
            lines.append(f"    Latest: {latest['timestamp'][:10]} ({latest['model_version']})")

        lines.append("=" * 60)
        return "\n".join(lines)
