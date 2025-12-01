#!/usr/bin/env python3
"""
ONNX Export Script with Quantization

Export trained models to ONNX format for optimized CPU/GPU inference.
Supports dynamic and static quantization for production deployment.

Features:
    - ONNX export with configurable opset
    - Dynamic INT8 quantization
    - Static INT8 quantization with calibration
    - FP16 conversion for GPU inference
    - Model validation after export
    - Size and accuracy comparison reports

Usage:
    # Basic ONNX export
    python export_utility/export_onnx.py \
        --model outputs/modernbert-multitask-v0 \
        --output outputs/onnx-export \
        --capability sentiment

    # Export with dynamic quantization (INT8)
    python export_utility/export_onnx.py \
        --model outputs/modernbert-multitask-v0 \
        --output outputs/onnx-quantized \
        --capability sentiment \
        --quantize dynamic

    # Export with static quantization (requires calibration data)
    python export_utility/export_onnx.py \
        --model outputs/modernbert-multitask-v0 \
        --output outputs/onnx-static-quant \
        --capability sentiment \
        --quantize static \
        --calibration-data data/calibration_samples.jsonl

    # Export all capabilities
    python export_utility/export_onnx.py \
        --model outputs/modernbert-multitask-v0 \
        --output outputs/onnx-all \
        --capability all

    # FP16 export for GPU
    python export_utility/export_onnx.py \
        --model outputs/modernbert-multitask-v0 \
        --output outputs/onnx-fp16 \
        --capability sentiment \
        --fp16

Requirements:
    - onnx
    - onnxruntime (or onnxruntime-gpu)
    - onnxruntime-tools (for quantization)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer

# =============================================================================
# Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default ONNX opset version
DEFAULT_OPSET = 17

# Supported quantization modes
QUANTIZATION_MODES = ["none", "dynamic", "static"]


# =============================================================================
# ONNX Export Wrapper
# =============================================================================


class ONNXExportWrapper(torch.nn.Module):
    """Wrapper module for ONNX export of a single capability."""

    def __init__(self, model, capability: str):
        super().__init__()
        self.model = model
        self.capability = capability
        self.is_sequence_labeling = capability in [
            "ner_general",
            "ner_family",
            "temporal",
        ]
        self.is_embedding = capability == "embedding"

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for ONNX export."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task=self.capability,
        )

        if self.is_embedding:
            return outputs.get("embeddings", outputs.get("logits"))
        else:
            return outputs["logits"]


# =============================================================================
# Export Functions
# =============================================================================


def export_to_onnx(
    model,
    tokenizer,
    capability: str,
    output_path: Path,
    opset_version: int = DEFAULT_OPSET,
    dynamic_axes: bool = True,
) -> Path:
    """Export model to ONNX format."""
    import onnx

    logger.info(f"Exporting {capability} to ONNX (opset {opset_version})...")

    # Create wrapper
    wrapper = ONNXExportWrapper(model, capability)
    wrapper.eval()

    # Create dummy input
    dummy_text = "This is a sample input for ONNX export."
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        max_length=128,
        padding="max_length",
        truncation=True,
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Define output path
    onnx_path = output_path / f"{capability}.onnx"

    # Dynamic axes for variable batch size and sequence length
    if dynamic_axes:
        dynamic_axes_config = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size"},
        }
        # For sequence labeling, output has sequence dimension
        if wrapper.is_sequence_labeling:
            dynamic_axes_config["output"][1] = "sequence_length"
        # For embeddings, output is [batch, hidden_size]
        if wrapper.is_embedding:
            dynamic_axes_config["output"] = {0: "batch_size"}
    else:
        dynamic_axes_config = None

    # Export
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (input_ids, attention_mask),
            str(onnx_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["output"],
            dynamic_axes=dynamic_axes_config,
        )

    # Validate ONNX model
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    logger.info(f"✅ Exported to {onnx_path}")
    return onnx_path


def apply_dynamic_quantization(
    onnx_path: Path,
    output_path: Path,
) -> Path:
    """Apply dynamic INT8 quantization to ONNX model."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantized_path = output_path.with_name(output_path.stem + "_quantized_dynamic.onnx")

    logger.info("Applying dynamic INT8 quantization...")

    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(quantized_path),
        weight_type=QuantType.QInt8,
    )

    logger.info(f"✅ Dynamic quantization complete: {quantized_path}")
    return quantized_path


def apply_static_quantization(
    onnx_path: Path,
    output_path: Path,
    calibration_data: list[dict],
    tokenizer,
) -> Path:
    """Apply static INT8 quantization with calibration data."""
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    class CalibrationDataReaderImpl(CalibrationDataReader):
        """Calibration data reader for static quantization."""

        def __init__(self, calibration_samples: list[dict], tokenizer, max_samples: int = 100):
            self.samples = calibration_samples[:max_samples]
            self.tokenizer = tokenizer
            self.index = 0

        def get_next(self) -> dict | None:
            if self.index >= len(self.samples):
                return None

            sample = self.samples[self.index]
            text = sample.get("text", sample.get("sentence", ""))

            inputs = self.tokenizer(
                text,
                return_tensors="np",
                max_length=128,
                padding="max_length",
                truncation=True,
            )

            self.index += 1

            return {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
            }

        def rewind(self):
            self.index = 0

    quantized_path = output_path.with_name(output_path.stem + "_quantized_static.onnx")

    logger.info(f"Applying static INT8 quantization with {len(calibration_data)} samples...")

    # Create calibration reader
    calibration_reader = CalibrationDataReaderImpl(calibration_data, tokenizer, max_samples=100)

    quantize_static(
        model_input=str(onnx_path),
        model_output=str(quantized_path),
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )

    logger.info(f"✅ Static quantization complete: {quantized_path}")
    return quantized_path


def convert_to_fp16(onnx_path: Path, output_path: Path) -> Path:
    """Convert ONNX model to FP16 for GPU inference."""
    import onnx
    from onnxconverter_common import float16

    fp16_path = output_path.with_name(output_path.stem + "_fp16.onnx")

    logger.info("Converting to FP16...")

    model = onnx.load(str(onnx_path))
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, str(fp16_path))

    logger.info(f"✅ FP16 conversion complete: {fp16_path}")
    return fp16_path


# =============================================================================
# Validation Functions
# =============================================================================


def validate_onnx_model(
    onnx_path: Path,
    pytorch_model,
    tokenizer,
    capability: str,
    num_samples: int = 10,
    tolerance: float = 1e-4,
) -> dict:
    """Validate ONNX model against PyTorch model."""
    import onnxruntime as ort

    logger.info(f"Validating ONNX model: {onnx_path.name}")

    # Create ONNX session
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    # Create wrapper for PyTorch
    wrapper = ONNXExportWrapper(pytorch_model, capability)
    wrapper.eval()

    # Test samples
    test_texts = [
        "This is a test sentence for validation.",
        "Another sample to check model consistency.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models need proper validation.",
        "ONNX export should preserve model accuracy.",
    ] * 2  # Repeat to get more samples

    errors = []
    max_diff = 0.0

    for text in test_texts[:num_samples]:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=128,
            padding="max_length",
            truncation=True,
        )

        # PyTorch inference
        with torch.no_grad():
            pt_output = wrapper(
                inputs["input_ids"],
                inputs["attention_mask"],
            ).numpy()

        # ONNX inference
        ort_inputs = {
            "input_ids": inputs["input_ids"].numpy().astype(np.int64),
            "attention_mask": inputs["attention_mask"].numpy().astype(np.int64),
        }
        ort_output = session.run(None, ort_inputs)[0]

        # Compare outputs
        diff = np.abs(pt_output - ort_output).max()
        max_diff = max(max_diff, diff)

        if diff > tolerance:
            errors.append({"text": text[:50], "max_diff": float(diff)})

    validation_passed = len(errors) == 0

    result = {
        "validated": validation_passed,
        "num_samples": num_samples,
        "max_difference": float(max_diff),
        "tolerance": tolerance,
        "errors": errors[:5] if errors else [],
    }

    if validation_passed:
        logger.info(f"✅ Validation passed (max diff: {max_diff:.6f})")
    else:
        logger.warning(f"⚠️ Validation failed with {len(errors)} errors (max diff: {max_diff:.6f})")

    return result


def benchmark_onnx_model(
    onnx_path: Path,
    tokenizer,
    batch_size: int = 1,
    seq_length: int = 128,
    num_iterations: int = 100,
) -> dict:
    """Benchmark ONNX model inference latency."""
    import onnxruntime as ort

    logger.info(f"Benchmarking ONNX model: {onnx_path.name}")

    # Create session with optimization
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    # Create dummy input
    dummy_text = "This is a benchmark test. " * (seq_length // 8)
    inputs = tokenizer(
        [dummy_text] * batch_size,
        return_tensors="np",
        max_length=seq_length,
        padding="max_length",
        truncation=True,
    )

    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }

    # Warmup
    for _ in range(10):
        _ = session.run(None, ort_inputs)

    # Benchmark
    latencies = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = session.run(None, ort_inputs)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms

    latencies.sort()
    n = len(latencies)

    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "p50_ms": float(latencies[int(n * 0.50)]),
        "p95_ms": float(latencies[int(n * 0.95)]),
        "p99_ms": float(latencies[min(int(n * 0.99), n - 1)]),
        "throughput_qps": float(1000.0 / np.mean(latencies)),
        "batch_size": batch_size,
        "seq_length": seq_length,
    }


def get_model_size(path: Path) -> float:
    """Get model file size in MB."""
    return path.stat().st_size / (1024 * 1024)


# =============================================================================
# Report Generation
# =============================================================================


def generate_export_report(
    results: dict,
    output_dir: Path,
) -> None:
    """Generate export report."""
    report_path = output_dir / "export_report.json"

    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Report saved to {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("ONNX EXPORT SUMMARY")
    print("=" * 60)

    for capability, cap_results in results.get("capabilities", {}).items():
        print(f"\n{capability}:")
        for model_type, info in cap_results.items():
            if isinstance(info, dict) and "size_mb" in info:
                print(f"  {model_type}:")
                print(f"    Size: {info['size_mb']:.2f} MB")
                if "benchmark" in info:
                    bm = info["benchmark"]
                    print(f"    Latency: {bm['mean_ms']:.2f}ms (p99: {bm['p99_ms']:.2f}ms)")
                    print(f"    Throughput: {bm['throughput_qps']:.1f} QPS")

    print("=" * 60)


# =============================================================================
# Main
# =============================================================================


def load_calibration_data(path: str | Path) -> list[dict]:
    """Load calibration data from JSONL file."""
    samples = []
    path = Path(path)

    if not path.exists():
        logger.warning(f"Calibration file not found: {path}")
        return samples

    with open(path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    logger.info(f"Loaded {len(samples)} calibration samples")
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Export model to ONNX format with optional quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic export
  python export_utility/export_onnx.py \\
      --model outputs/modernbert-multitask-v0 \\
      --output outputs/onnx \\
      --capability sentiment

  # With dynamic quantization
  python export_utility/export_onnx.py \\
      --model outputs/modernbert-multitask-v0 \\
      --output outputs/onnx-quant \\
      --capability sentiment \\
      --quantize dynamic

  # Export all capabilities
  python export_utility/export_onnx.py \\
      --model outputs/modernbert-multitask-v0 \\
      --output outputs/onnx-all \\
      --capability all
        """,
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Path to model directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output directory for ONNX models",
    )
    parser.add_argument(
        "--capability",
        "-c",
        type=str,
        nargs="+",
        default=["sentiment"],
        help="Capabilities to export (or 'all')",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=DEFAULT_OPSET,
        help=f"ONNX opset version (default: {DEFAULT_OPSET})",
    )
    parser.add_argument(
        "--quantize",
        "-q",
        type=str,
        choices=QUANTIZATION_MODES,
        default="none",
        help="Quantization mode (none, dynamic, static)",
    )
    parser.add_argument(
        "--calibration-data",
        type=str,
        default=None,
        help="Path to calibration data JSONL (for static quantization)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Convert to FP16 for GPU inference",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate exported model against PyTorch",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run inference benchmark on exported models",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip model validation",
    )

    args = parser.parse_args()

    # Validate paths
    model_path = Path(args.model)
    output_path = Path(args.output)

    if not model_path.exists():
        logger.error(f"Model path does not exist: {model_path}")
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

    logger.info(f"Loading model from {model_path}")
    model = ModernBertMultiTaskModel.from_pretrained(str(model_path))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    logger.info(f"Model loaded with heads: {list(model.heads.keys())}")

    # Resolve capabilities
    if "all" in args.capability:
        capabilities = list(model.heads.keys())
    else:
        capabilities = [c for c in args.capability if c in model.heads]

    if not capabilities:
        logger.error("No valid capabilities to export")
        sys.exit(1)

    logger.info(f"Exporting capabilities: {capabilities}")

    # Load calibration data if needed
    calibration_data = None
    if args.quantize == "static":
        if not args.calibration_data:
            logger.error("Static quantization requires --calibration-data")
            sys.exit(1)
        calibration_data = load_calibration_data(args.calibration_data)
        if not calibration_data:
            logger.error("No calibration data loaded")
            sys.exit(1)

    # Export each capability
    results = {
        "model_path": str(model_path),
        "output_path": str(output_path),
        "opset_version": args.opset,
        "quantization": args.quantize,
        "fp16": args.fp16,
        "capabilities": {},
    }

    for capability in capabilities:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing: {capability}")
        logger.info(f"{'='*50}")

        cap_results = {}

        # Export base ONNX model
        onnx_path = export_to_onnx(model, tokenizer, capability, output_path, args.opset)
        cap_results["base"] = {
            "path": str(onnx_path),
            "size_mb": get_model_size(onnx_path),
        }

        # Validate
        if not args.skip_validation:
            validation = validate_onnx_model(onnx_path, model, tokenizer, capability)
            cap_results["base"]["validation"] = validation

        # Benchmark
        if args.benchmark:
            benchmark = benchmark_onnx_model(onnx_path, tokenizer)
            cap_results["base"]["benchmark"] = benchmark

        # Apply quantization
        if args.quantize == "dynamic":
            quant_path = apply_dynamic_quantization(onnx_path, onnx_path)
            cap_results["quantized_dynamic"] = {
                "path": str(quant_path),
                "size_mb": get_model_size(quant_path),
            }
            if args.benchmark:
                cap_results["quantized_dynamic"]["benchmark"] = benchmark_onnx_model(
                    quant_path, tokenizer
                )

        elif args.quantize == "static" and calibration_data:
            quant_path = apply_static_quantization(
                onnx_path, onnx_path, calibration_data, tokenizer
            )
            cap_results["quantized_static"] = {
                "path": str(quant_path),
                "size_mb": get_model_size(quant_path),
            }
            if args.benchmark:
                cap_results["quantized_static"]["benchmark"] = benchmark_onnx_model(
                    quant_path, tokenizer
                )

        # Convert to FP16
        if args.fp16:
            try:
                fp16_path = convert_to_fp16(onnx_path, onnx_path)
                cap_results["fp16"] = {
                    "path": str(fp16_path),
                    "size_mb": get_model_size(fp16_path),
                }
            except Exception as e:
                logger.warning(f"FP16 conversion failed: {e}")

        results["capabilities"][capability] = cap_results

    # Generate report
    generate_export_report(results, output_path)

    # Calculate size reduction
    if args.quantize != "none":
        print("\nSize Reduction Summary:")
        for cap, cap_results in results["capabilities"].items():
            base_size = cap_results["base"]["size_mb"]
            quant_key = f"quantized_{args.quantize}"
            if quant_key in cap_results:
                quant_size = cap_results[quant_key]["size_mb"]
                reduction = (1 - quant_size / base_size) * 100
                print(
                    f"  {cap}: {base_size:.1f}MB → {quant_size:.1f}MB ({reduction:.1f}% reduction)"
                )

    logger.info("\n✅ ONNX export complete!")


if __name__ == "__main__":
    main()
