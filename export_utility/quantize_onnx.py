#!/usr/bin/env python3
"""
ONNX Quantization Utility

Quantize ONNX models for edge deployment with multiple precision options:
- FP16: Half precision for GPU inference (2x smaller, ~1.5x faster)
- INT8 Dynamic: Integer quantization without calibration (4x smaller, ~2x faster)
- INT8 Static: Integer quantization with calibration (4x smaller, ~3x faster)

Features:
    - Dynamic INT8 quantization (no calibration data needed)
    - Static INT8 quantization (better accuracy with calibration)
    - FP16 conversion for GPU inference
    - Batch quantization of all ONNX files in a directory
    - Validation of quantized models

Usage:
    # Dynamic INT8 quantization (recommended for most cases)
    python quantize_onnx.py \\
        --input exports/decoder-onnx-v3/decoder_core.onnx \\
        --output exports/decoder-onnx-v3/decoder_core_int8.onnx \\
        --mode dynamic

    # Quantize all ONNX files in directory
    python quantize_onnx.py \\
        --input-dir exports/decoder-onnx-v3 \\
        --output-dir exports/decoder-onnx-v3-int8 \\
        --mode dynamic

    # FP16 conversion for GPU
    python quantize_onnx.py \\
        --input exports/decoder-onnx-v3/decoder_core.onnx \\
        --output exports/decoder-onnx-v3/decoder_core_fp16.onnx \\
        --mode fp16

    # Static INT8 with calibration data
    python quantize_onnx.py \\
        --input exports/decoder-onnx-v3/decoder_core.onnx \\
        --output exports/decoder-onnx-v3/decoder_core_int8_static.onnx \\
        --mode static \\
        --calibration-data data/calibration_samples.jsonl

Requirements:
    - onnx
    - onnxruntime
    - onnxruntime-tools (for quantization)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Iterator

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Quantization Functions
# =============================================================================


def quantize_dynamic_int8(
    input_path: Path,
    output_path: Path,
    per_channel: bool = False,
) -> Path:
    """
    Apply dynamic INT8 quantization to ONNX model.

    Dynamic quantization quantizes weights statically but activations dynamically
    at runtime. No calibration data is needed.

    Args:
        input_path: Path to input ONNX model
        output_path: Path to output quantized model
        per_channel: Use per-channel quantization (better accuracy, slightly slower)

    Returns:
        Path to quantized model
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    logger.info(f"Quantizing {input_path.name} with dynamic INT8...")

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        reduce_range=False,  # Set True for older CPUs without VNNI
    )

    input_size = input_path.stat().st_size / (1024 * 1024)
    output_size = output_path.stat().st_size / (1024 * 1024)
    reduction = (1 - output_size / input_size) * 100

    logger.info(f"  Input:  {input_size:.2f} MB")
    logger.info(f"  Output: {output_size:.2f} MB ({reduction:.1f}% reduction)")

    return output_path


class CalibrationDataReader:
    """
    Calibration data reader for static INT8 quantization.

    Provides sample inputs to measure activation ranges for optimal quantization.
    """

    def __init__(
        self,
        calibration_data: List[dict],
        input_names: List[str],
    ):
        self.calibration_data = calibration_data
        self.input_names = input_names
        self.current_index = 0

    def get_next(self) -> Optional[dict]:
        """Get next calibration sample."""
        if self.current_index >= len(self.calibration_data):
            return None

        sample = self.calibration_data[self.current_index]
        self.current_index += 1

        # Convert to numpy arrays
        result = {}
        for name in self.input_names:
            if name in sample:
                result[name] = np.array(sample[name], dtype=np.float32)

        return result

    def rewind(self):
        """Reset to beginning of calibration data."""
        self.current_index = 0


def generate_calibration_data(
    model_path: Path,
    num_samples: int = 100,
) -> List[dict]:
    """
    Generate synthetic calibration data for a model.

    For production, use real representative data from your dataset.

    Args:
        model_path: Path to ONNX model
        num_samples: Number of calibration samples to generate

    Returns:
        List of input dictionaries
    """
    import onnx

    logger.info(f"Generating {num_samples} synthetic calibration samples...")

    model = onnx.load(str(model_path))

    calibration_data = []
    for _ in range(num_samples):
        sample = {}
        for input_info in model.graph.input:
            name = input_info.name
            shape = []

            for dim in input_info.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                else:
                    # Dynamic dimension - use reasonable default
                    shape.append(16)

            # Generate random data based on expected type
            elem_type = input_info.type.tensor_type.elem_type
            if elem_type == 1:  # FLOAT
                sample[name] = np.random.randn(*shape).astype(np.float32).tolist()
            elif elem_type == 7:  # INT64
                sample[name] = np.random.randint(0, 100, shape).astype(np.int64).tolist()
            else:
                sample[name] = np.random.randn(*shape).astype(np.float32).tolist()

        calibration_data.append(sample)

    return calibration_data


def quantize_static_int8(
    input_path: Path,
    output_path: Path,
    calibration_data: Optional[List[dict]] = None,
    calibration_file: Optional[Path] = None,
    per_channel: bool = True,
) -> Path:
    """
    Apply static INT8 quantization to ONNX model.

    Static quantization quantizes both weights and activations statically,
    using calibration data to determine optimal quantization parameters.

    Args:
        input_path: Path to input ONNX model
        output_path: Path to output quantized model
        calibration_data: List of calibration samples (dicts of input arrays)
        calibration_file: Path to JSONL file with calibration samples
        per_channel: Use per-channel quantization

    Returns:
        Path to quantized model
    """
    from onnxruntime.quantization import quantize_static, QuantType, CalibrationMethod
    import onnx

    logger.info(f"Quantizing {input_path.name} with static INT8...")

    # Load calibration data
    if calibration_data is None:
        if calibration_file is not None and calibration_file.exists():
            logger.info(f"Loading calibration data from {calibration_file}")
            calibration_data = []
            with open(calibration_file) as f:
                for line in f:
                    calibration_data.append(json.loads(line))
        else:
            # Generate synthetic calibration data
            calibration_data = generate_calibration_data(input_path)

    # Get input names
    model = onnx.load(str(input_path))
    input_names = [inp.name for inp in model.graph.input]

    # Create calibration reader
    reader = CalibrationDataReader(calibration_data, input_names)

    # Quantize
    quantize_static(
        model_input=str(input_path),
        model_output=str(output_path),
        calibration_data_reader=reader,
        quant_format=None,  # Use default
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        calibrate_method=CalibrationMethod.MinMax,
    )

    input_size = input_path.stat().st_size / (1024 * 1024)
    output_size = output_path.stat().st_size / (1024 * 1024)
    reduction = (1 - output_size / input_size) * 100

    logger.info(f"  Input:  {input_size:.2f} MB")
    logger.info(f"  Output: {output_size:.2f} MB ({reduction:.1f}% reduction)")

    return output_path


def convert_to_fp16(
    input_path: Path,
    output_path: Path,
    keep_io_types: bool = True,
) -> Path:
    """
    Convert ONNX model to FP16 (half precision).

    FP16 provides 2x memory reduction with minimal accuracy loss.
    Best for GPU inference where FP16 is natively supported.

    Args:
        input_path: Path to input ONNX model
        output_path: Path to output FP16 model
        keep_io_types: Keep input/output as FP32 for compatibility

    Returns:
        Path to FP16 model
    """
    from onnxconverter_common import float16
    import onnx

    logger.info(f"Converting {input_path.name} to FP16...")

    model = onnx.load(str(input_path))

    # Convert to FP16
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=keep_io_types,
        disable_shape_infer=False,
    )

    onnx.save(model_fp16, str(output_path))

    input_size = input_path.stat().st_size / (1024 * 1024)
    output_size = output_path.stat().st_size / (1024 * 1024)
    reduction = (1 - output_size / input_size) * 100

    logger.info(f"  Input:  {input_size:.2f} MB")
    logger.info(f"  Output: {output_size:.2f} MB ({reduction:.1f}% reduction)")

    return output_path


def quantize_directory(
    input_dir: Path,
    output_dir: Path,
    mode: str = "dynamic",
    calibration_file: Optional[Path] = None,
) -> List[Path]:
    """
    Quantize all ONNX models in a directory.

    Args:
        input_dir: Directory containing ONNX models
        output_dir: Output directory for quantized models
        mode: Quantization mode ("dynamic", "static", or "fp16")
        calibration_file: Path to calibration data (for static mode)

    Returns:
        List of paths to quantized models
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    quantized_paths = []

    for onnx_file in input_dir.glob("*.onnx"):
        # Generate output filename
        stem = onnx_file.stem
        if mode == "fp16":
            output_name = f"{stem}_fp16.onnx"
        elif mode == "static":
            output_name = f"{stem}_int8_static.onnx"
        else:
            output_name = f"{stem}_int8.onnx"

        output_path = output_dir / output_name

        # Quantize
        if mode == "fp16":
            convert_to_fp16(onnx_file, output_path)
        elif mode == "static":
            quantize_static_int8(onnx_file, output_path, calibration_file=calibration_file)
        else:
            quantize_dynamic_int8(onnx_file, output_path)

        quantized_paths.append(output_path)

    return quantized_paths


def validate_quantized_model(
    original_path: Path,
    quantized_path: Path,
    rtol: float = 0.1,
    atol: float = 0.1,
) -> bool:
    """
    Validate quantized model produces similar outputs to original.

    Note: INT8 quantization may have larger numerical differences than FP32.
    The tolerances here are relaxed accordingly.

    Args:
        original_path: Path to original ONNX model
        quantized_path: Path to quantized model
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        True if validation passes
    """
    import onnxruntime as ort
    import onnx

    logger.info(f"Validating {quantized_path.name}...")

    # Load models
    original_session = ort.InferenceSession(
        str(original_path),
        providers=["CPUExecutionProvider"],
    )
    quantized_session = ort.InferenceSession(
        str(quantized_path),
        providers=["CPUExecutionProvider"],
    )

    # Get input info
    model = onnx.load(str(original_path))

    # Generate test input
    test_inputs = {}
    for input_info in model.graph.input:
        name = input_info.name
        shape = []

        for dim in input_info.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            else:
                shape.append(4)  # Small batch for testing

        elem_type = input_info.type.tensor_type.elem_type
        if elem_type == 1:  # FLOAT
            test_inputs[name] = np.random.randn(*shape).astype(np.float32)
        elif elem_type == 7:  # INT64
            test_inputs[name] = np.random.randint(0, 100, shape).astype(np.int64)
        else:
            test_inputs[name] = np.random.randn(*shape).astype(np.float32)

    # Run both models
    original_outputs = original_session.run(None, test_inputs)
    quantized_outputs = quantized_session.run(None, test_inputs)

    # Compare outputs
    all_close = True
    for i, (orig, quant) in enumerate(zip(original_outputs, quantized_outputs)):
        if not np.allclose(orig, quant, rtol=rtol, atol=atol):
            max_diff = np.max(np.abs(orig - quant))
            mean_diff = np.mean(np.abs(orig - quant))
            logger.warning(f"  Output {i}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
            all_close = False

    if all_close:
        logger.info(f"  Validation PASSED")
    else:
        logger.warning(f"  Validation FAILED (outputs differ beyond tolerance)")
        logger.warning(f"  This may be acceptable for INT8 quantization")

    return all_close


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="ONNX Quantization Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quantization Modes:
    dynamic - INT8 dynamic quantization (no calibration needed, ~4x smaller)
    static  - INT8 static quantization (requires calibration, ~4x smaller, best accuracy)
    fp16    - FP16 half precision (2x smaller, best for GPU)

Examples:
    # Dynamic INT8 (recommended)
    python quantize_onnx.py \\
        --input decoder_core.onnx \\
        --output decoder_core_int8.onnx \\
        --mode dynamic

    # Quantize entire directory
    python quantize_onnx.py \\
        --input-dir exports/decoder-onnx-v3 \\
        --output-dir exports/decoder-onnx-v3-int8 \\
        --mode dynamic

    # FP16 for GPU
    python quantize_onnx.py \\
        --input decoder_core.onnx \\
        --output decoder_core_fp16.onnx \\
        --mode fp16
        """,
    )

    parser.add_argument(
        "--input",
        type=str,
        help="Path to input ONNX model",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output quantized model",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing ONNX models to quantize",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for quantized models",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="dynamic",
        choices=["dynamic", "static", "fp16"],
        help="Quantization mode (default: dynamic)",
    )
    parser.add_argument(
        "--calibration-data",
        type=str,
        help="Path to calibration data JSONL file (for static mode)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate quantized model outputs",
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="Use per-channel quantization (better accuracy)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.input_dir:
        # Directory mode
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir) if args.output_dir else input_dir.parent / f"{input_dir.name}_{args.mode}"

        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            sys.exit(1)

        calibration_file = Path(args.calibration_data) if args.calibration_data else None

        logger.info("=" * 60)
        logger.info(f"Quantizing directory: {input_dir}")
        logger.info(f"Mode: {args.mode}")
        logger.info(f"Output: {output_dir}")
        logger.info("=" * 60)

        quantized_paths = quantize_directory(
            input_dir,
            output_dir,
            mode=args.mode,
            calibration_file=calibration_file,
        )

        # Validate if requested
        if args.validate:
            logger.info("\nValidating quantized models...")
            for quant_path in quantized_paths:
                # Find original
                orig_name = quant_path.stem.replace("_int8", "").replace("_int8_static", "").replace("_fp16", "")
                orig_path = input_dir / f"{orig_name}.onnx"
                if orig_path.exists():
                    validate_quantized_model(orig_path, quant_path)

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("Summary:")
        logger.info("=" * 60)

        total_input = sum(
            (input_dir / f.stem.replace("_int8", "").replace("_int8_static", "").replace("_fp16", "") + ".onnx").stat().st_size
            for f in quantized_paths
            if (input_dir / (f.stem.replace("_int8", "").replace("_int8_static", "").replace("_fp16", "") + ".onnx")).exists()
        ) / (1024 * 1024)
        total_output = sum(f.stat().st_size for f in quantized_paths) / (1024 * 1024)

        logger.info(f"  Files quantized: {len(quantized_paths)}")
        logger.info(f"  Total output size: {total_output:.2f} MB")

    elif args.input:
        # Single file mode
        input_path = Path(args.input)

        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            sys.exit(1)

        if args.output:
            output_path = Path(args.output)
        else:
            suffix = "_int8" if args.mode in ["dynamic", "static"] else "_fp16"
            output_path = input_path.parent / f"{input_path.stem}{suffix}.onnx"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info(f"Quantizing: {input_path.name}")
        logger.info(f"Mode: {args.mode}")
        logger.info("=" * 60)

        # Quantize
        if args.mode == "fp16":
            convert_to_fp16(input_path, output_path)
        elif args.mode == "static":
            calibration_file = Path(args.calibration_data) if args.calibration_data else None
            quantize_static_int8(
                input_path,
                output_path,
                calibration_file=calibration_file,
                per_channel=args.per_channel,
            )
        else:
            quantize_dynamic_int8(
                input_path,
                output_path,
                per_channel=args.per_channel,
            )

        # Validate if requested
        if args.validate:
            validate_quantized_model(input_path, output_path)

        logger.info(f"\nOutput: {output_path}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
